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

// --- MCP Server setup ---
const server = new McpServer({
  name: "task-external",
  version: "1.0.0",
});

server.tool(
  "Task",
  "外部の信頼できるスーパーエージェントを起動し、複雑なマルチステップタスクを自律的に処理させます。",
  {
    description: z.string().describe("A short (3-5 word) description of the task"),
    prompt: z.string().describe("The task for the agent to perform"),
    subagent_type: z
      .string()
      .describe(
        'Agent type: "general-purpose" (full access), "Explore" (read-only), "Plan" (read-only)'
      ),
    model: z
      .string()
      .optional()
      .describe("Optional model to use (e.g. codex-mini, o4-mini, o3)"),
    mode: z
      .string()
      .optional()
      .describe(
        'Permission mode: "default", "acceptEdits", "dontAsk", "bypassPermissions", "plan"'
      ),
    isolation: z
      .enum(["worktree"])
      .optional()
      .describe('Set to "worktree" for isolated working directory'),
    run_in_background: z
      .boolean()
      .optional()
      .describe("Set to true to run in background (not yet supported)"),
    resume: z
      .string()
      .optional()
      .describe("Thread ID from a previous invocation to resume"),
  },
  async (params) => {
    const {
      description,
      prompt,
      subagent_type,
      model,
      mode,
      isolation,
      resume,
    } = params;

    console.error(
      `[task-external] Starting task: "${description}" (type=${subagent_type}, model=${model ?? "default"}, mode=${mode ?? "default"})`
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

      if (model) {
        threadOptions.model = model;
      }

      if (isolation === "worktree") {
        // Use a temporary directory for isolation
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
        DEFAULT_TIMEOUT_MS
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
      console.error(`[task-external] Task failed: "${description}" - ${errorMessage}`);

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
  }
);

function formatTurnResult(turn: RunResult, threadId: string | null): string {
  const parts: string[] = [];

  // Final response text
  if (turn.finalResponse) {
    parts.push(turn.finalResponse);
  }

  // Summary of items (file changes, commands, etc.)
  const fileChanges = turn.items.filter((i: ThreadItem) => i.type === "file_change");
  const commands = turn.items.filter((i: ThreadItem) => i.type === "command_execution");

  if (fileChanges.length > 0) {
    parts.push(
      `\n[Files changed: ${fileChanges.length}]`
    );
  }
  if (commands.length > 0) {
    parts.push(
      `[Commands executed: ${commands.length}]`
    );
  }

  // Include threadId for resume capability
  if (threadId) {
    parts.push(`\n[agentId: ${threadId}]`);
  }

  // Usage info
  if (turn.usage) {
    parts.push(
      `[Tokens - input: ${turn.usage.input_tokens}, output: ${turn.usage.output_tokens}]`
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
