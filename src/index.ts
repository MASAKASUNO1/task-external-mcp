#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  Codex,
  type ThreadOptions,
  type ApprovalMode,
  type SandboxMode,
  type RunResult,
  type ThreadItem,
} from "@openai/codex-sdk";
import { z } from "zod";

// --- Mapping tables ---

const SUBAGENT_TYPE_MAP: Record<
  string,
  { sandboxMode: SandboxMode; approvalPolicy: ApprovalMode }
> = {
  "general-purpose": {
    sandboxMode: "workspace-write",
    approvalPolicy: "on-request",
  },
  Explore: { sandboxMode: "read-only", approvalPolicy: "never" },
  Plan: { sandboxMode: "read-only", approvalPolicy: "never" },
};

const MODE_TO_APPROVAL: Record<string, ApprovalMode> = {
  default: "on-request",
  acceptEdits: "on-request",
  dontAsk: "never",
  bypassPermissions: "never",
  plan: "untrusted",
};

const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

// --- Codex singleton ---
const codex = new Codex();

// --- Tool description (matches Claude Code's Task tool) ---
const TOOL_DESCRIPTION = `外部の信頼できるスーパーエージェントを起動し、複雑なマルチステップタスクを自律的に処理させます。

The Task tool launches specialized agents that autonomously handle complex tasks.

Available agent types:
- general-purpose: General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. (Tools: *)
- Explore: Fast agent specialized for exploring codebases. Use for quick file searches, keyword searches, or codebase questions. (Tools: read-only)
- Plan: Software architect agent for designing implementation plans. Returns step-by-step plans and considers architectural trade-offs. (Tools: read-only)

Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do
- Launch multiple agents concurrently whenever possible, to maximize performance
- When the agent is done, it will return a single message back to you with the result
- Provide clear, detailed prompts so the agent can work autonomously
- Use run_in_background for genuinely independent work
- Agents can be resumed using the resume parameter by passing the agent ID from a previous invocation`;

// --- MCP Server setup ---
const server = new McpServer({
  name: "task-external",
  version: "1.0.0",
});

server.tool(
  "Task",
  TOOL_DESCRIPTION,
  {
    description: z
      .string()
      .describe("A short (3-5 word) description of the task"),
    prompt: z
      .string()
      .describe("The task for the agent to perform"),
    subagent_type: z
      .string()
      .describe("The type of specialized agent to use for this task"),
    model: z
      .enum(["sonnet", "opus", "haiku"])
      .optional()
      .describe(
        "Optional model to use for this agent. If not specified, inherits from parent. Prefer haiku for quick, straightforward tasks to minimize cost and latency."
      ),
    mode: z
      .enum(["acceptEdits", "bypassPermissions", "default", "dontAsk", "plan"])
      .optional()
      .describe(
        'Permission mode for spawned teammate (e.g., "plan" to require plan approval).'
      ),
    isolation: z
      .enum(["worktree"])
      .optional()
      .describe(
        'Isolation mode. "worktree" creates a temporary working directory so the agent works on an isolated copy.'
      ),
    max_turns: z
      .number()
      .int()
      .positive()
      .optional()
      .describe(
        "Maximum number of agentic turns (API round-trips) before stopping."
      ),
    name: z
      .string()
      .optional()
      .describe("Name for the spawned agent"),
    run_in_background: z
      .boolean()
      .optional()
      .describe(
        "Set to true to run this agent in the background. The result will be returned when the agent completes."
      ),
    resume: z
      .string()
      .optional()
      .describe(
        "Optional agent ID to resume from. If provided, the agent will continue from the previous execution."
      ),
    team_name: z
      .string()
      .optional()
      .describe(
        "Team name for spawning. Uses current team context if omitted."
      ),
  },
  async (params) => {
    const {
      description,
      prompt,
      subagent_type,
      model,
      mode,
      isolation,
      name,
      resume,
    } = params;

    console.error(
      `[task-external] Starting task: "${description}" (type=${subagent_type}, model=${model ?? "default"}, mode=${mode ?? "default"}, name=${name ?? "anonymous"})`
    );

    try {
      // Resolve sandbox/approval from subagent_type
      const subagentConfig = SUBAGENT_TYPE_MAP[subagent_type] ?? {
        sandboxMode: "workspace-write" as SandboxMode,
        approvalPolicy: "on-request" as ApprovalMode,
      };

      // Mode override for approval policy
      const approvalPolicy: ApprovalMode =
        mode && MODE_TO_APPROVAL[mode]
          ? MODE_TO_APPROVAL[mode]!
          : subagentConfig.approvalPolicy;

      // Build thread options
      const threadOptions: ThreadOptions = {
        sandboxMode: subagentConfig.sandboxMode,
        approvalPolicy,
        skipGitRepoCheck: true,
      };

      // Map model enum to Codex model string
      if (model) {
        const MODEL_MAP: Record<string, string> = {
          opus: "o3",
          sonnet: "o4-mini",
          haiku: "codex-mini",
        };
        threadOptions.model = MODEL_MAP[model] ?? model;
      }

      if (isolation === "worktree") {
        const { mkdtemp } = await import("node:fs/promises");
        const { tmpdir } = await import("node:os");
        const { join } = await import("node:path");
        const workDir = await mkdtemp(join(tmpdir(), "task-external-"));
        threadOptions.workingDirectory = workDir;
        console.error(`[task-external] Worktree isolation: ${workDir}`);
      }

      // Create or resume thread
      const thread = resume
        ? codex.resumeThread(resume, threadOptions)
        : codex.startThread(threadOptions);

      // Run with timeout
      const controller = new AbortController();
      const timeout = setTimeout(
        () => controller.abort(),
        DEFAULT_TIMEOUT_MS,
      );

      let turn: RunResult;
      try {
        turn = await thread.run(prompt, { signal: controller.signal });
      } finally {
        clearTimeout(timeout);
      }

      const threadId = thread.id;

      console.error(
        `[task-external] Task completed: "${description}" (threadId=${threadId})`
      );

      // Format response
      const responseText = formatTurnResult(turn, threadId);

      return {
        content: [{ type: "text" as const, text: responseText }],
      };
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      console.error(
        `[task-external] Task failed: "${description}" - ${errorMessage}`
      );

      return {
        content: [
          {
            type: "text" as const,
            text: `Task "${description}" failed: ${errorMessage}`,
          },
        ],
        isError: true,
      };
    }
  },
);

function formatTurnResult(turn: RunResult, threadId: string | null): string {
  const parts: string[] = [];

  if (turn.finalResponse) {
    parts.push(turn.finalResponse);
  }

  const fileChanges = turn.items.filter(
    (i: ThreadItem) => i.type === "file_change",
  );
  const commands = turn.items.filter(
    (i: ThreadItem) => i.type === "command_execution",
  );

  if (fileChanges.length > 0) {
    parts.push(`\n[Files changed: ${fileChanges.length}]`);
  }
  if (commands.length > 0) {
    parts.push(`[Commands executed: ${commands.length}]`);
  }

  if (threadId) {
    parts.push(`\n[agentId: ${threadId}]`);
  }

  if (turn.usage) {
    parts.push(
      `[Tokens - input: ${turn.usage.input_tokens}, output: ${turn.usage.output_tokens}]`,
    );
  }

  return parts.join("\n");
}

// --- Start server ---
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[task-external] MCP server started on stdio");
}

main().catch((err) => {
  console.error("[task-external] Fatal error:", err);
  process.exit(1);
});
