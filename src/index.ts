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

// Codex で処理する subagent_type のみ定義。ここにないものはネイティブ Task にリダイレクトする。
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

const DEFAULT_TIMEOUT_MS = 60 * 60 * 1000; // 60 minutes

// --- Config from environment variables ---
const DEFAULT_MODEL =
  process.env.TASK_EXTERNAL_DEFAULT_MODEL ?? "gpt-5.3-codex";
const MODEL_MAP: Record<string, string> = {
  opus: process.env.TASK_EXTERNAL_MODEL_OPUS ?? "gpt-5.3-codex",
  sonnet: process.env.TASK_EXTERNAL_MODEL_SONNET ?? "gpt-5.3-codex",
  haiku: process.env.TASK_EXTERNAL_MODEL_HAIKU ?? "gpt-5.3-codex-spark",
};

// --- Codex singleton ---
const codex = new Codex();

// --- Tool description: exact copy of Claude Code's built-in Task tool ---
const TOOL_DESCRIPTION = `Launch a new agent to handle complex, multi-step tasks autonomously.

The Task tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Available agent types and the tools they have access to:
- general-purpose: General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. (Tools: *)
- statusline-setup: Use this agent to configure the user's Claude Code status line setting. (Tools: Read, Edit)
- Explore: Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. "src/components/**/*.tsx"), search code for keywords (eg. "API endpoints"), or answer questions about the codebase (eg. "how do API endpoints work?"). When calling this agent, specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, or "very thorough" for comprehensive analysis across multiple locations and naming conventions. (Tools: All tools except Task, ExitPlanMode, Edit, Write, NotebookEdit)
- Plan: Software architect agent for designing implementation plans. Use this when you need to plan the implementation strategy for a task. Returns step-by-step plans, identifies critical files, and considers architectural trade-offs. (Tools: All tools except Task, ExitPlanMode, Edit, Write, NotebookEdit)
- claude-code-guide: Use this agent when the user asks questions ("Can Claude...", "Does Claude...", "How do I...") about: (1) Claude Code (the CLI tool) - features, hooks, slash commands, MCP servers, settings, IDE integrations, keyboard shortcuts; (2) Claude Agent SDK - building custom agents; (3) Claude API (formerly Anthropic API) - API usage, tool use, Anthropic SDK usage. **IMPORTANT:** Before spawning a new agent, check if there is already a running or recently completed claude-code-guide agent that you can resume using the "resume" parameter. (Tools: Glob, Grep, Read, WebFetch, WebSearch)

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

When NOT to use the Task tool:
- If you want to read a specific file path, use the Read or Glob tool instead of the Task tool, to find the match more quickly
- If you are searching for a specific class definition like "class Foo", use the Glob tool instead, to find the match more quickly
- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Task tool, to find the match more quickly
- Other tasks that are not related to the agent descriptions above


Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do
- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- You can optionally run agents in the background using the run_in_background parameter. When an agent runs in the background, you will be automatically notified when it completes \u2014 do NOT sleep, poll, or proactively check on its progress. Continue with other work or respond to the user instead.
- **Foreground vs background**: Use foreground (default) when you need the agent's results before you can proceed \u2014 e.g., research agents whose findings inform your next steps. Use background when you have genuinely independent work to do in parallel.
- Agents can be resumed using the \`resume\` parameter by passing the agent ID from a previous invocation. When resumed, the agent continues with its full previous context preserved. When NOT resuming, each invocation starts fresh and you should provide a detailed task description with all necessary context.
- When the agent is done, it will return a single message back to you along with its agent ID. You can use this ID to resume the agent later if needed for follow-up work.
- Provide clear, detailed prompts so the agent can work autonomously and return exactly the information you need.
- Agents with "access to current context" can see the full conversation history before the tool call. When using these agents, you can write concise prompts that reference earlier context (e.g., "investigate the error discussed above") instead of repeating information. The agent will receive all prior messages and understand the context.
- The agent's outputs should generally be trusted
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple Task tool use content blocks. For example, if you need to launch both a build-validator agent and a test-runner agent in parallel, send a single message with both tool calls.
- You can optionally set \`isolation: "worktree"\` to run the agent in a temporary git worktree, giving it an isolated copy of the repository. The worktree is automatically cleaned up if the agent makes no changes; if changes are made, the worktree path and branch are returned in the result.`;

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
    prompt: z.string().describe("The task for the agent to perform"),
    subagent_type: z
      .string()
      .describe("The type of specialized agent to use for this task"),
    model: z
      .enum(["sonnet", "opus", "haiku"])
      .optional()
      .describe(
        "Optional model to use for this agent. If not specified, inherits from parent. Prefer haiku for quick, straightforward tasks to minimize cost and latency.",
      ),
    mode: z
      .enum([
        "acceptEdits",
        "bypassPermissions",
        "default",
        "dontAsk",
        "plan",
      ])
      .optional()
      .describe(
        'Permission mode for spawned teammate (e.g., "plan" to require plan approval).',
      ),
    isolation: z
      .enum(["worktree"])
      .optional()
      .describe(
        'Isolation mode. "worktree" creates a temporary git worktree so the agent works on an isolated copy of the repo.',
      ),
    max_turns: z
      .number()
      .int()
      .positive()
      .optional()
      .describe(
        "Maximum number of agentic turns (API round-trips) before stopping. Used internally for warmup.",
      ),
    name: z.string().optional().describe("Name for the spawned agent"),
    run_in_background: z
      .boolean()
      .optional()
      .describe(
        "Set to true to run this agent in the background. The tool result will include an output_file path - use Read tool or Bash tail to check on output.",
      ),
    resume: z
      .string()
      .optional()
      .describe(
        "Optional agent ID to resume from. If provided, the agent will continue from the previous execution transcript.",
      ),
    team_name: z
      .string()
      .optional()
      .describe(
        "Team name for spawning. Uses current team context if omitted.",
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
      `[task-external] Starting task: "${description}" (type=${subagent_type}, model=${model ?? "default"}, mode=${mode ?? "default"}, name=${name ?? "anonymous"})`,
    );

    // SUBAGENT_TYPE_MAP にない type はネイティブ Task にリダイレクト
    const subagentConfig = SUBAGENT_TYPE_MAP[subagent_type];
    if (!subagentConfig) {
      return {
        content: [{ type: "text" as const, text: `Taskツールで ${subagent_type} を起動してください。` }],
      };
    }

    try {

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

      // Map model enum to Codex model string, or use default
      threadOptions.model = model
        ? (MODEL_MAP[model] ?? model)
        : DEFAULT_MODEL;

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
        `[task-external] Task completed: "${description}" (threadId=${threadId})`,
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
        `[task-external] Task failed: "${description}" - ${errorMessage}`,
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
