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
import { readFile, writeFile, mkdir, unlink } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { open } from "node:fs/promises";
import { join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { execFile as execFileCb } from "node:child_process";
import { promisify } from "node:util";
import { z } from "zod";

const execFile = promisify(execFileCb);

// --- Custom agent definition loader ---

interface AgentDefinition {
  name: string;
  description?: string;
  tools?: string;
  model?: string;
  permissionMode?: string;
  maxTurns?: number;
  isolation?: string;
  systemPrompt: string; // markdown body
}

/**
 * .claude/agents/{name}.md を探して読み込む。
 * 探索順: プロジェクト (.claude/agents/) → ユーザー (~/.claude/agents/)
 */
async function loadAgentDefinition(
  name: string,
): Promise<AgentDefinition | null> {
  const candidates = [
    join(process.cwd(), ".claude", "agents", `${name}.md`),
    join(homedir(), ".claude", "agents", `${name}.md`),
  ];

  for (const filePath of candidates) {
    try {
      const content = await readFile(filePath, "utf-8");
      return parseAgentFile(content, name);
    } catch {
      // file not found, try next
    }
  }
  return null;
}

function parseAgentFile(content: string, fallbackName: string): AgentDefinition {
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!frontmatterMatch) {
    return { name: fallbackName, systemPrompt: content.trim() };
  }

  const yamlBlock = frontmatterMatch[1]!;
  const body = frontmatterMatch[2]!.trim();

  // Simple YAML key-value parser (no nested objects)
  const meta: Record<string, string> = {};
  for (const line of yamlBlock.split("\n")) {
    const match = line.match(/^(\w+)\s*:\s*(.+)$/);
    if (match) {
      meta[match[1]!] = match[2]!.trim();
    }
  }

  return {
    name: meta["name"] ?? fallbackName,
    description: meta["description"],
    tools: meta["tools"],
    model: meta["model"],
    permissionMode: meta["permissionMode"],
    maxTurns: meta["maxTurns"] ? Number(meta["maxTurns"]) : undefined,
    isolation: meta["isolation"],
    systemPrompt: body,
  };
}

// --- Git worktree helpers ---

interface WorktreeInfo {
  path: string;
  branch: string;
  originHead: string;
}

async function createWorktree(): Promise<WorktreeInfo> {
  const timestamp = Date.now();
  const branch = `task-external/${timestamp}`;
  const worktreePath = join(tmpdir(), `task-external-wt-${timestamp}`);

  await execFile("git", ["worktree", "add", worktreePath, "-b", branch]);
  const { stdout: headSha } = await execFile("git", ["rev-parse", "HEAD"], {
    cwd: worktreePath,
  });

  console.error(
    `[task-external] Created git worktree: path=${worktreePath}, branch=${branch}`,
  );

  return { path: worktreePath, branch, originHead: headSha.trim() };
}

async function cleanupWorktree(wt: WorktreeInfo): Promise<{ hasChanges: boolean }> {
  try {
    const { stdout: status } = await execFile(
      "git",
      ["status", "--porcelain"],
      { cwd: wt.path },
    );
    const { stdout: currentHead } = await execFile(
      "git",
      ["rev-parse", "HEAD"],
      { cwd: wt.path },
    );
    const hasChanges =
      status.trim().length > 0 || currentHead.trim() !== wt.originHead;

    if (!hasChanges) {
      await execFile("git", ["worktree", "remove", "--force", wt.path]);
      try {
        await execFile("git", ["branch", "-D", wt.branch]);
      } catch {
        // branch may already be deleted
      }
      console.error(
        `[task-external] Worktree cleaned up (no changes): ${wt.path}`,
      );
    } else {
      console.error(
        `[task-external] Worktree retained (changes detected): ${wt.path} on branch ${wt.branch}`,
      );
    }

    return { hasChanges };
  } catch (err) {
    console.error(
      `[task-external] Error during worktree cleanup: ${err instanceof Error ? err.message : err}`,
    );
    // Force remove on error to prevent leaks
    try {
      await execFile("git", ["worktree", "remove", "--force", wt.path]);
    } catch {
      // best effort
    }
    return { hasChanges: false };
  }
}

// --- Team infrastructure types ---

interface TeamMember {
  name: string;
  agentId: string;
  agentType: string;
}

interface TeamConfig {
  team_name: string;
  description?: string;
  members: TeamMember[];
}

interface InboxMessage {
  id: string;
  from: string;
  content: string;
  summary?: string;
  timestamp: string;
  read: boolean;
  type?: string;         // "message" | "shutdown_request" | etc.
  request_id?: string;   // for shutdown_request
}

type TeamMessage =
  | { type: "message"; recipient: string; content: string; summary: string }
  | { type: "broadcast"; content: string; summary: string }
  | { type: "shutdown_response"; request_id: string; approve: boolean; content?: string };

// --- Team path utilities ---

function teamDir(teamName: string): string {
  return join(homedir(), ".claude", "teams", teamName);
}

function inboxDir(teamName: string, agentName: string): string {
  return join(homedir(), ".claude", "teams", teamName, "inboxes", agentName);
}

function tasksDir(teamName: string): string {
  return join(homedir(), ".claude", "tasks", teamName);
}

// --- Team file I/O layer ---

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function readTeamConfig(teamName: string): Promise<TeamConfig> {
  try {
    const raw = await readFile(join(teamDir(teamName), "config.json"), "utf-8");
    return JSON.parse(raw) as TeamConfig;
  } catch {
    return { team_name: teamName, members: [] };
  }
}

async function writeTeamConfig(
  teamName: string,
  config: TeamConfig,
): Promise<void> {
  const dir = teamDir(teamName);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "config.json"), JSON.stringify(config, null, 2));
}

async function withTeamLock<T>(
  teamName: string,
  fn: () => Promise<T>,
): Promise<T> {
  const lockPath = join(teamDir(teamName), ".lock");
  await mkdir(teamDir(teamName), { recursive: true });

  let fd: import("node:fs/promises").FileHandle | null = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      fd = await open(lockPath, fsConstants.O_CREAT | fsConstants.O_EXCL | fsConstants.O_WRONLY);
      break;
    } catch {
      await sleep(100);
    }
  }
  if (!fd) {
    throw new Error(`[task-external] Failed to acquire team lock for ${teamName}`);
  }

  try {
    return await fn();
  } finally {
    await fd.close();
    await unlink(lockPath).catch(() => {});
  }
}

async function addMember(
  teamName: string,
  member: TeamMember,
): Promise<void> {
  await withTeamLock(teamName, async () => {
    const config = await readTeamConfig(teamName);
    if (!config.members.some((m) => m.name === member.name)) {
      config.members.push(member);
      await writeTeamConfig(teamName, config);
    }
  });
}

async function removeMember(
  teamName: string,
  agentName: string,
): Promise<void> {
  await withTeamLock(teamName, async () => {
    const config = await readTeamConfig(teamName);
    config.members = config.members.filter((m) => m.name !== agentName);
    await writeTeamConfig(teamName, config);
  });
}

async function appendInboxMessage(
  teamName: string,
  agentName: string,
  msg: Omit<InboxMessage, "id" | "timestamp" | "read">,
): Promise<void> {
  const dir = inboxDir(teamName, agentName);
  await mkdir(dir, { recursive: true });
  const filePath = join(dir, "messages.json");

  let messages: InboxMessage[] = [];
  try {
    const raw = await readFile(filePath, "utf-8");
    messages = JSON.parse(raw) as InboxMessage[];
  } catch {
    // file doesn't exist yet
  }

  const full: InboxMessage = {
    ...msg,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
    read: false,
  };
  messages.push(full);
  await writeFile(filePath, JSON.stringify(messages, null, 2));

  console.error(
    `[task-external] Inbox message appended: ${teamName}/${agentName} from=${msg.from}`,
  );
}

async function drainInbox(
  teamName: string,
  agentName: string,
): Promise<InboxMessage[]> {
  const filePath = join(inboxDir(teamName, agentName), "messages.json");

  let messages: InboxMessage[] = [];
  try {
    const raw = await readFile(filePath, "utf-8");
    messages = JSON.parse(raw) as InboxMessage[];
  } catch {
    return [];
  }

  const unread = messages.filter((m) => !m.read);
  if (unread.length === 0) return [];

  // Mark as read
  for (const m of unread) {
    m.read = true;
  }
  await writeFile(filePath, JSON.stringify(messages, null, 2));
  return unread;
}

async function pollInbox(
  teamName: string,
  agentName: string,
  signal?: AbortSignal,
): Promise<InboxMessage[]> {
  const MAX_POLLS = 150; // 150 * 2s = 300s
  for (let i = 0; i < MAX_POLLS; i++) {
    if (signal?.aborted) return [];
    const msgs = await drainInbox(teamName, agentName);
    if (msgs.length > 0) return msgs;
    await sleep(2000);
  }
  console.error(
    `[task-external] pollInbox timeout: ${teamName}/${agentName}`,
  );
  return [];
}

// --- Team message parsing + prompt injection ---

const TEAM_AGENT_SYSTEM_PROMPT_SUFFIX = `
You are operating as a teammate in a collaborative agent team.

To send messages to other agents, include markers in your output using this exact format:

<!--SEND_MESSAGE
{"type":"message","recipient":"agent-name","content":"Your message here","summary":"Brief summary"}
-->

To broadcast to all teammates:
<!--SEND_MESSAGE
{"type":"broadcast","content":"Message to all","summary":"Brief summary"}
-->

When you receive a shutdown_request, respond with:
<!--SEND_MESSAGE
{"type":"shutdown_response","request_id":"the-request-id","approve":true}
-->

You can place these markers anywhere in your output. Any text outside the markers will be treated as your response to the team leader.

IMPORTANT: Always use the <!--SEND_MESSAGE ... --> format for inter-agent communication. Plain text output alone will not be delivered to other agents.
`.trim();

const SEND_MESSAGE_RE = /<!--SEND_MESSAGE\n([\s\S]*?)\n-->/g;

function parseAgentOutput(output: string): {
  cleanText: string;
  messages: TeamMessage[];
  isShuttingDown: boolean;
} {
  const messages: TeamMessage[] = [];
  let isShuttingDown = false;

  const cleanText = output.replace(SEND_MESSAGE_RE, (_, jsonStr: string) => {
    try {
      const parsed = JSON.parse(jsonStr);
      if (parsed.type === "message" && parsed.recipient && parsed.content) {
        messages.push({
          type: "message",
          recipient: parsed.recipient,
          content: parsed.content,
          summary: parsed.summary ?? "",
        });
      } else if (parsed.type === "broadcast" && parsed.content) {
        messages.push({
          type: "broadcast",
          content: parsed.content,
          summary: parsed.summary ?? "",
        });
      } else if (parsed.type === "shutdown_response") {
        messages.push({
          type: "shutdown_response",
          request_id: parsed.request_id,
          approve: !!parsed.approve,
          content: parsed.content,
        });
        if (parsed.approve) {
          isShuttingDown = true;
        }
      }
    } catch (err) {
      console.error(
        `[task-external] Failed to parse SEND_MESSAGE marker: ${err instanceof Error ? err.message : err}`,
      );
    }
    return "";
  }).trim();

  return { cleanText, messages, isShuttingDown };
}

const AGENT_COLORS = [
  "\x1b[36m", // cyan
  "\x1b[33m", // yellow
  "\x1b[35m", // magenta
  "\x1b[32m", // green
  "\x1b[34m", // blue
  "\x1b[91m", // bright red
  "\x1b[96m", // bright cyan
  "\x1b[93m", // bright yellow
];
const RESET_COLOR = "\x1b[0m";
const agentColorMap = new Map<string, string>();

function resolveAgentColor(agentName: string): string {
  let color = agentColorMap.get(agentName);
  if (!color) {
    color = AGENT_COLORS[agentColorMap.size % AGENT_COLORS.length]!;
    agentColorMap.set(agentName, color);
  }
  return color;
}

// --- Mapping tables ---

// Codex で直接処理する組み込み subagent_type。
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
- You can optionally set \`isolation: "worktree"\` to run the agent in a temporary git worktree, giving it an isolated copy of the repository. The worktree is automatically cleaned up if the agent makes no changes; if changes are made, the worktree path and branch are returned in the result.

IMPORTANT - Team mode (team_name parameter):
When you specify team_name and name, the agent runs as a long-lived teammate in the background. However, this tool uses file-based inboxes with NO push notifications. You MUST call the CheckTeamInbox tool to receive messages from teammates. Without calling CheckTeamInbox, teammate outputs will never reach you. After spawning team agents, immediately call CheckTeamInbox(team_name=..., block=true) to wait for their results.`;

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
      max_turns,
      name,
      run_in_background,
      resume,
      team_name,
    } = params;

    console.error(
      `[task-external] Starting task: "${description}" (type=${subagent_type}, model=${model ?? "default"}, mode=${mode ?? "default"}, name=${name ?? "anonymous"}, team=${team_name ?? "none"})`,
    );
    if (max_turns != null) {
      console.error(
        `[task-external] max_turns=${max_turns} requested but Codex SDK does not support maxTurns — parameter accepted for compatibility`,
      );
    }

    // 1. 組み込み type を確認
    const builtinConfig = SUBAGENT_TYPE_MAP[subagent_type];

    // 2. 組み込みにない場合、カスタムエージェント定義を探す
    let agentDef: AgentDefinition | null = null;
    if (!builtinConfig) {
      agentDef = await loadAgentDefinition(subagent_type);
      if (!agentDef) {
        // どちらにもない → ネイティブ Task にリダイレクト
        return {
          content: [{ type: "text" as const, text: `Taskツールで ${subagent_type} を起動してください。` }],
        };
      }
      console.error(
        `[task-external] Loaded custom agent definition: ${subagent_type}`,
      );
    }

    try {
      // sandbox/approval を決定
      let sandboxMode: SandboxMode;
      let baseApproval: ApprovalMode;

      if (builtinConfig) {
        sandboxMode = builtinConfig.sandboxMode;
        baseApproval = builtinConfig.approvalPolicy;
      } else {
        // カスタムエージェント: permissionMode から推定、デフォルトは general-purpose 相当
        const permMode = agentDef!.permissionMode ?? mode ?? "default";
        sandboxMode = "workspace-write";
        baseApproval = MODE_TO_APPROVAL[permMode] ?? "on-request";
      }

      // Mode override for approval policy
      const approvalPolicy: ApprovalMode =
        mode && MODE_TO_APPROVAL[mode]
          ? MODE_TO_APPROVAL[mode]!
          : baseApproval;

      // Build thread options
      const threadOptions: ThreadOptions = {
        sandboxMode,
        approvalPolicy,
        skipGitRepoCheck: true,
        webSearchMode: "live",
        webSearchEnabled: true,
      };

      // Model: パラメータ > エージェント定義 > デフォルト
      if (model) {
        threadOptions.model = MODEL_MAP[model] ?? model;
      } else if (agentDef?.model && MODEL_MAP[agentDef.model]) {
        threadOptions.model = MODEL_MAP[agentDef.model]!;
      } else {
        threadOptions.model = DEFAULT_MODEL;
      }

      // プロンプト構築: カスタムエージェントの場合は system prompt を先頭に注入
      let finalPrompt = prompt;
      if (agentDef) {
        finalPrompt =
          `<system>\n${agentDef.systemPrompt}\n</system>\n\n${prompt}`;
      }

      const effectiveIsolation = isolation ?? agentDef?.isolation;

      // --- Team mode: fire-and-forget with message loop ---
      if (team_name && name) {
        const agentName = name;

        // Fire-and-forget: launch background team agent loop
        runTeamAgent({
          teamName: team_name,
          agentName,
          agentType: subagent_type,
          prompt: finalPrompt,
          threadOptions,
          description,
        }).catch((err) => {
          console.error(
            `[task-external] Team agent "${agentName}" crashed: ${err instanceof Error ? err.message : err}`,
          );
        });

        console.error(
          `[task-external] Team agent spawned: ${agentName}@${team_name}`,
        );

        return {
          content: [
            {
              type: "text" as const,
              text: [
                `Spawned successfully.`,
                `agent_id: ${agentName}@${team_name}`,
                `name: ${agentName}`,
                `team_name: ${team_name}`,
                ``,
                `IMPORTANT: This agent runs in the background but CANNOT send push notifications.`,
                `You MUST call the CheckTeamInbox tool to receive messages from this agent.`,
                `Recommended workflow:`,
                `  1. Spawn all agents you need`,
                `  2. Call CheckTeamInbox(team_name="${team_name}", block=true, timeout=120000) to wait for results`,
                `  3. Process received messages and repeat CheckTeamInbox if more agents are still working`,
                `Without calling CheckTeamInbox, you will never receive the agent's output.`,
              ].join("\n"),
            },
          ],
        };
      }

      // --- Background requests are executed in foreground ---
      // MCP tools cannot run in background (Claude Code closes the
      // connection on background responses). Always run synchronously
      // and let Claude Code wait for the result.
      if (run_in_background) {
        console.error(
          `[task-external] run_in_background requested but executing in foreground (MCP limitation): "${description}"`,
        );
      }

      // --- Foreground execution path ---
      let worktree: WorktreeInfo | null = null;
      if (effectiveIsolation === "worktree") {
        worktree = await createWorktree();
        threadOptions.workingDirectory = worktree.path;
      }

      try {
        const thread = resume
          ? codex.resumeThread(resume, threadOptions)
          : codex.startThread(threadOptions);

        const controller = new AbortController();
        const timeoutHandle = setTimeout(
          () => controller.abort(),
          DEFAULT_TIMEOUT_MS,
        );

        let turn: RunResult;
        try {
          turn = await thread.run(finalPrompt, { signal: controller.signal });
        } finally {
          clearTimeout(timeoutHandle);
        }

        const threadId = thread.id;

        console.error(
          `[task-external] Task completed: "${description}" (threadId=${threadId})`,
        );

        // Worktree cleanup: check for changes
        let worktreePath: string | undefined;
        let worktreeBranch: string | undefined;
        if (worktree) {
          const { hasChanges } = await cleanupWorktree(worktree);
          if (hasChanges) {
            worktreePath = worktree.path;
            worktreeBranch = worktree.branch;
          }
        }

        // Format response
        let responseText = "";
        if (run_in_background) {
          responseText += "Note: run_in_background was requested, but this tool does not support background execution. The task was executed in foreground instead.\n\n";
        }
        responseText += formatTurnResult(turn, {
          threadId,
          worktreePath,
          worktreeBranch,
          name,
          teamName: team_name,
        });

        return {
          content: [{ type: "text" as const, text: responseText }],
        };
      } catch (innerError) {
        // Ensure worktree is cleaned up even on Codex execution failure
        if (worktree) {
          await cleanupWorktree(worktree);
        }
        throw innerError;
      }
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

// --- Team agent message loop ---

interface RunTeamAgentParams {
  teamName: string;
  agentName: string;
  agentType: string;
  prompt: string;
  threadOptions: ThreadOptions;
  description: string;
}

async function runTeamAgent(params: RunTeamAgentParams): Promise<void> {
  const { teamName, agentName, agentType, prompt, threadOptions, description } =
    params;
  const color = resolveAgentColor(agentName);
  const log = (msg: string) =>
    console.error(`${color}[team:${teamName}/${agentName}]${RESET_COLOR} ${msg}`);

  // 1. Register member in config.json
  const agentId = `${agentName}@${teamName}`;
  await addMember(teamName, { name: agentName, agentId, agentType });
  log(`Registered as member (agentId=${agentId})`);

  const thread = codex.startThread(threadOptions);

  try {
    // 2. First turn with TEAM_AGENT_SYSTEM_PROMPT_SUFFIX injected
    const fullPrompt = `${TEAM_AGENT_SYSTEM_PROMPT_SUFFIX}\n\n---\n\nYour name is "${agentName}" and you are on team "${teamName}".\n\n${prompt}`;
    let currentInput = fullPrompt;
    let turnCount = 0;

    while (true) {
      turnCount++;
      log(`Starting turn ${turnCount}...`);

      const controller = new AbortController();
      const timeoutHandle = setTimeout(
        () => controller.abort(),
        DEFAULT_TIMEOUT_MS,
      );

      let turn: RunResult;
      try {
        turn = await thread.run(currentInput, { signal: controller.signal });
      } finally {
        clearTimeout(timeoutHandle);
      }

      const output = turn.finalResponse ?? "";
      log(`Turn ${turnCount} completed (${output.length} chars)`);

      // 3. Parse output for markers
      const { cleanText, messages, isShuttingDown } = parseAgentOutput(output);

      // 4. Deliver messages
      for (const msg of messages) {
        if (msg.type === "message") {
          await appendInboxMessage(teamName, msg.recipient, {
            from: agentName,
            content: msg.content,
            summary: msg.summary,
            type: "message",
          });
          log(`Sent message to ${msg.recipient}`);
        } else if (msg.type === "broadcast") {
          // Send to all members except self
          const config = await readTeamConfig(teamName);
          for (const member of config.members) {
            if (member.name !== agentName) {
              await appendInboxMessage(teamName, member.name, {
                from: agentName,
                content: msg.content,
                summary: msg.summary,
                type: "message",
              });
            }
          }
          log(`Broadcast sent to ${config.members.length - 1} members`);
        } else if (msg.type === "shutdown_response" && msg.approve) {
          // Send shutdown_approved to team-lead
          await appendInboxMessage(teamName, "team-lead", {
            from: agentName,
            content: JSON.stringify({
              type: "shutdown_approved",
              agent: agentName,
            }),
            summary: `${agentName} shutdown approved`,
            type: "shutdown_approved",
          });
          log("Shutdown approved, exiting loop");
        }
      }

      // 5. If shutting down, break out of loop
      if (isShuttingDown) {
        log("Agent shutting down");
        break;
      }

      // 6. Send idle_notification to team-lead
      if (cleanText) {
        await appendInboxMessage(teamName, "team-lead", {
          from: agentName,
          content: cleanText,
          summary: `${agentName} completed turn ${turnCount}`,
          type: "idle_notification",
        });
      } else {
        await appendInboxMessage(teamName, "team-lead", {
          from: agentName,
          content: JSON.stringify({
            type: "idle",
            agent: agentName,
            turn: turnCount,
          }),
          summary: `${agentName} is idle after turn ${turnCount}`,
          type: "idle_notification",
        });
      }

      // 7. Poll inbox for next message
      log("Polling inbox for next message...");
      const inboxMessages = await pollInbox(teamName, agentName);
      if (inboxMessages.length === 0) {
        log("Poll timeout, exiting loop");
        break;
      }

      // Check for shutdown_request
      const shutdownReq = inboxMessages.find(
        (m) => m.type === "shutdown_request",
      );
      if (shutdownReq) {
        // Pass shutdown to the agent so it can cleanly respond
        currentInput = `You have received a shutdown request (request_id: "${shutdownReq.request_id ?? shutdownReq.id}"). Please wrap up and respond with a shutdown_response marker to approve the shutdown:\n<!--SEND_MESSAGE\n{"type":"shutdown_response","request_id":"${shutdownReq.request_id ?? shutdownReq.id}","approve":true}\n-->`;
        continue;
      }

      // Concatenate all messages as next input
      currentInput = inboxMessages
        .map(
          (m) =>
            `[Message from ${m.from}${m.type ? ` (${m.type})` : ""}]:\n${m.content}`,
        )
        .join("\n\n---\n\n");
    }

    log(`Agent loop ended after ${turnCount} turns`);
  } catch (err) {
    log(
      `Error: ${err instanceof Error ? err.message : String(err)}`,
    );
  } finally {
    // Clean up: remove member from config
    await removeMember(teamName, agentName).catch((e) =>
      log(`Failed to remove member: ${e}`),
    );
    log("Member removed from config");
  }
}

interface FormatOptions {
  threadId: string | null;
  worktreePath?: string;
  worktreeBranch?: string;
  name?: string;
  teamName?: string;
}

function formatTurnResult(turn: RunResult, options: FormatOptions): string {
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

  if (options.worktreePath && options.worktreeBranch) {
    parts.push(`\n[worktree_path: ${options.worktreePath}]`);
    parts.push(`[worktree_branch: ${options.worktreeBranch}]`);
  }

  if (options.threadId) {
    parts.push(`\n[agentId: ${options.threadId}]`);
  }
  if (options.name) {
    parts.push(`[agent_name: ${options.name}]`);
  }
  if (options.teamName) {
    parts.push(`[team_name: ${options.teamName}]`);
  }

  if (turn.usage) {
    parts.push(
      `[Tokens - input: ${turn.usage.input_tokens}, output: ${turn.usage.output_tokens}]`,
    );
  }

  return parts.join("\n");
}

// --- SendMessage tool ---

server.tool(
  "SendMessage",
  "Send a message to a specific agent or broadcast to all agents in a team. Use this to coordinate between team members.",
  {
    type: z
      .enum(["message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"])
      .describe("Message type"),
    recipient: z
      .string()
      .optional()
      .describe("Target agent name (required for message, shutdown_request, plan_approval_response)"),
    content: z
      .string()
      .optional()
      .describe("Message body"),
    summary: z
      .string()
      .optional()
      .describe("Short summary (5-10 words) shown as preview"),
    request_id: z
      .string()
      .optional()
      .describe("Request ID to respond to (required for shutdown_response, plan_approval_response)"),
    approve: z
      .boolean()
      .optional()
      .describe("Whether to approve (for shutdown_response, plan_approval_response)"),
    team_name: z
      .string()
      .describe("Team name the sender belongs to"),
    sender_name: z
      .string()
      .describe("Name of the sending agent"),
  },
  async (params) => {
    const { type, recipient, content, summary, request_id, approve, team_name, sender_name } = params;

    console.error(
      `[task-external] SendMessage: type=${type}, sender=${sender_name}, team=${team_name}, recipient=${recipient ?? "(broadcast)"}`,
    );

    try {
      if (type === "message") {
        if (!recipient) {
          return {
            content: [{ type: "text" as const, text: "Error: recipient is required for type=message" }],
            isError: true,
          };
        }
        await appendInboxMessage(team_name, recipient, {
          from: sender_name,
          content: content ?? "",
          summary: summary ?? "",
          type: "message",
        });
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ success: true, message: `Message sent to ${recipient}'s inbox`, routing: { sender: sender_name, target: `@${recipient}`, summary: summary ?? "" } }) }],
        };

      } else if (type === "broadcast") {
        const config = await readTeamConfig(team_name);
        let delivered = 0;
        for (const member of config.members) {
          if (member.name !== sender_name) {
            await appendInboxMessage(team_name, member.name, {
              from: sender_name,
              content: content ?? "",
              summary: summary ?? "",
              type: "broadcast",
            });
            delivered++;
          }
        }
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ success: true, message: `Broadcast delivered to ${delivered} teammates`, routing: { sender: sender_name, target: "@all", summary: summary ?? "" } }) }],
        };

      } else if (type === "shutdown_request") {
        if (!recipient) {
          return {
            content: [{ type: "text" as const, text: "Error: recipient is required for type=shutdown_request" }],
            isError: true,
          };
        }
        const reqId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        await appendInboxMessage(team_name, recipient, {
          from: sender_name,
          content: content ?? `Shutdown requested by ${sender_name}`,
          summary: summary ?? `Shutdown request from ${sender_name}`,
          type: "shutdown_request",
          request_id: reqId,
        });
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ success: true, message: `Shutdown request sent to ${recipient}`, request_id: reqId }) }],
        };

      } else if (type === "shutdown_response") {
        if (!request_id) {
          return {
            content: [{ type: "text" as const, text: "Error: request_id is required for type=shutdown_response" }],
            isError: true,
          };
        }
        // Notify team-lead of shutdown decision
        await appendInboxMessage(team_name, "team-lead", {
          from: sender_name,
          content: JSON.stringify({ type: "shutdown_response", request_id, approve: !!approve, content }),
          summary: `${sender_name} ${approve ? "approved" : "rejected"} shutdown`,
          type: "shutdown_response",
          request_id,
        });
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ success: true, message: `Shutdown response sent (approve=${approve})`, request_id }) }],
        };

      } else if (type === "plan_approval_response") {
        if (!recipient || !request_id) {
          return {
            content: [{ type: "text" as const, text: "Error: recipient and request_id are required for type=plan_approval_response" }],
            isError: true,
          };
        }
        await appendInboxMessage(team_name, recipient, {
          from: sender_name,
          content: JSON.stringify({ type: "plan_approval_response", request_id, approve: !!approve, content }),
          summary: `Plan ${approve ? "approved" : "rejected"} by ${sender_name}`,
          type: "plan_approval_response",
          request_id,
        });
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ success: true, message: `Plan approval response sent to ${recipient} (approve=${approve})`, request_id }) }],
        };
      }

      return {
        content: [{ type: "text" as const, text: `Error: Unknown message type: ${type}` }],
        isError: true,
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[task-external] SendMessage failed: ${msg}`);
      return {
        content: [{ type: "text" as const, text: `SendMessage failed: ${msg}` }],
        isError: true,
      };
    }
  },
);

// --- TeamCreate tool ---
// Creates team directory structure so teammates can coordinate.

server.tool(
  "TeamCreate",
  "Create a new team to coordinate multiple agents. Creates team config and task directories.",
  {
    team_name: z.string().describe("Name for the new team"),
    description: z.string().optional().describe("Team description/purpose"),
  },
  async (params) => {
    const { team_name, description } = params;

    console.error(`[task-external] TeamCreate: ${team_name}`);

    const dir = teamDir(team_name);
    await mkdir(dir, { recursive: true });
    await mkdir(join(dir, "inboxes", "team-lead"), { recursive: true });
    await mkdir(tasksDir(team_name), { recursive: true });

    const config: TeamConfig = {
      team_name,
      description,
      members: [{ name: "team-lead", agentId: `team-lead@${team_name}`, agentType: "leader" }],
    };
    await writeTeamConfig(team_name, config);

    return {
      content: [{
        type: "text" as const,
        text: [
          JSON.stringify({
            team_name,
            team_file_path: join(dir, "config.json"),
            lead_agent_id: `team-lead@${team_name}`,
          }, null, 2),
          ``,
          `Team "${team_name}" created successfully.`,
          ``,
          `=== HOW TEAM COMMUNICATION WORKS ===`,
          `This MCP-based team uses file-based inboxes. There are NO push notifications.`,
          `You (team-lead) MUST actively poll for messages using the CheckTeamInbox tool.`,
          ``,
          `Workflow:`,
          `  1. TeamCreate (done)`,
          `  2. Spawn agents with Task(team_name="${team_name}", name="agent-name", ...)`,
          `  3. After spawning, call CheckTeamInbox(team_name="${team_name}", block=true) to WAIT for results`,
          `  4. Process messages, send follow-ups with SendMessage if needed`,
          `  5. Repeat CheckTeamInbox until all work is done`,
          `  6. Shutdown agents with SendMessage(type="shutdown_request")`,
          `  7. TeamDelete to clean up`,
          ``,
          `CRITICAL: If you do not call CheckTeamInbox, you will NEVER receive agent outputs.`,
        ].join("\n"),
      }],
    };
  },
);

// --- TeamDelete tool ---

server.tool(
  "TeamDelete",
  "Remove team and task directories when the swarm work is complete.",
  {
    team_name: z.string().describe("Team name to delete"),
  },
  async (params) => {
    const { team_name } = params;

    console.error(`[task-external] TeamDelete: ${team_name}`);

    const config = await readTeamConfig(team_name);
    const activeMembers = config.members.filter((m) => m.name !== "team-lead");
    if (activeMembers.length > 0) {
      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({
            success: false,
            message: `Cannot cleanup team with ${activeMembers.length} active member(s): ${activeMembers.map((m) => m.name).join(", ")}. Use SendMessage with type=shutdown_request to gracefully terminate teammates first.`,
            team_name,
          }, null, 2),
        }],
      };
    }

    // Best-effort cleanup
    const { rm } = await import("node:fs/promises");
    try {
      await rm(teamDir(team_name), { recursive: true, force: true });
    } catch { /* best effort */ }
    try {
      await rm(tasksDir(team_name), { recursive: true, force: true });
    } catch { /* best effort */ }

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          success: true,
          message: `Cleaned up directories for team "${team_name}"`,
          team_name,
        }, null, 2),
      }],
    };
  },
);

// --- CheckTeamInbox tool ---
// Allows the team-lead (Claude Code) to receive messages from teammates.
// This is the critical missing piece: without this, the team-lead has
// no way to see teammate outputs, idle notifications, or shutdown responses.

server.tool(
  "CheckTeamInbox",
  `Check your inbox for messages from teammates. Use this to receive results, idle notifications, and status updates from team agents.

- Call with block=true to wait for new messages (up to timeout ms)
- Call with block=false to check immediately and return
- Returns all unread messages from teammates
- After spawning team agents, call this periodically to get their updates`,
  {
    team_name: z.string().describe("Team name"),
    agent_name: z
      .string()
      .optional()
      .default("team-lead")
      .describe("Your agent name (defaults to team-lead)"),
    block: z
      .boolean()
      .optional()
      .default(true)
      .describe("Whether to wait for new messages"),
    timeout: z
      .number()
      .optional()
      .default(60000)
      .describe("Max wait time in ms (default 60s, max 300s)"),
  },
  async (params) => {
    const { team_name, agent_name, block, timeout } = params;
    const effectiveAgent = agent_name ?? "team-lead";
    const effectiveTimeout = Math.min(timeout ?? 60000, 300000);

    console.error(
      `[task-external] CheckTeamInbox: team=${team_name}, agent=${effectiveAgent}, block=${block}, timeout=${effectiveTimeout}`,
    );

    if (!block) {
      // Non-blocking: return immediately
      const msgs = await drainInbox(team_name, effectiveAgent);
      if (msgs.length === 0) {
        return {
          content: [{
            type: "text" as const,
            text: "No new messages.",
          }],
        };
      }
      return {
        content: [{
          type: "text" as const,
          text: formatInboxMessages(msgs),
        }],
      };
    }

    // Blocking: poll until messages arrive or timeout
    const deadline = Date.now() + effectiveTimeout;
    while (Date.now() < deadline) {
      const msgs = await drainInbox(team_name, effectiveAgent);
      if (msgs.length > 0) {
        return {
          content: [{
            type: "text" as const,
            text: formatInboxMessages(msgs),
          }],
        };
      }
      await sleep(2000);
    }

    return {
      content: [{
        type: "text" as const,
        text: `No messages received after ${effectiveTimeout}ms.`,
      }],
    };
  },
);

function formatInboxMessages(msgs: InboxMessage[]): string {
  const parts: string[] = [];
  for (const m of msgs) {
    const header = `[From: ${m.from}] [Type: ${m.type ?? "message"}] [${m.timestamp}]`;
    if (m.summary) {
      parts.push(`${header}\nSummary: ${m.summary}\n${m.content}`);
    } else {
      parts.push(`${header}\n${m.content}`);
    }
  }
  return parts.join("\n\n---\n\n");
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
