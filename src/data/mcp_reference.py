"""Curated MCP reference-server tool definitions.

Provenance: transcribed from the public documentation / source of the official
Model Context Protocol reference servers (github.com/modelcontextprotocol/servers
and its archived servers: filesystem, git, github, fetch, memory, sqlite, time,
brave-search, puppeteer, slack, google-maps, everything, sequential-thinking).
Schemas reflect the documented ``tools/list`` contracts at the time of
transcription and may differ in minor details from the latest upstream versions.
They are labelled ``source = "mcp_reference_servers"`` in the dataset.

Run ``python -m src.data.mcp_reference`` to (re)write
``data/external/mcp_reference_tools.json``.
"""
from __future__ import annotations

from typing import Any

from src.utils.config import load_config, resolve
from src.utils.io import write_json


def S(desc: str | None = None, **kw) -> dict:
    d = {"type": "string"}
    if desc:
        d["description"] = desc
    d.update(kw)
    return d


def N(desc: str | None = None, t: str = "number", **kw) -> dict:
    d = {"type": t}
    if desc:
        d["description"] = desc
    d.update(kw)
    return d


def B(desc: str | None = None, **kw) -> dict:
    d = {"type": "boolean"}
    if desc:
        d["description"] = desc
    d.update(kw)
    return d


def A(items: dict, desc: str | None = None, **kw) -> dict:
    d = {"type": "array", "items": items}
    if desc:
        d["description"] = desc
    d.update(kw)
    return d


def O(props: dict, required: list[str] | None = None, desc: str | None = None) -> dict:
    d: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        d["required"] = required
    if desc:
        d["description"] = desc
    return d


def T(server: str, name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"server_name": server, "tool_name": name, "description": description,
            "input_schema": O(props, required)}


PATH = S("Path to the file or directory")

TOOLS: list[dict] = [
    # ------------------------------------------------------------- filesystem
    T("filesystem", "read_file", "Read the complete contents of a file from the file system. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Only works within allowed directories.",
      {"path": PATH}, ["path"]),
    T("filesystem", "read_multiple_files", "Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.",
      {"paths": A(S(), "List of file paths to read")}, ["paths"]),
    T("filesystem", "write_file", "Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.",
      {"path": PATH, "content": S("Text content to write")}, ["path", "content"]),
    T("filesystem", "edit_file", "Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style diff showing the changes made. Only works within allowed directories.",
      {"path": PATH, "edits": A(O({"oldText": S("Text to search for - must match exactly"), "newText": S("Text to replace with")}, ["oldText", "newText"])),
       "dryRun": B("Preview changes using git-style diff format", default=False)}, ["path", "edits"]),
    T("filesystem", "create_directory", "Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Only works within allowed directories.",
      {"path": PATH}, ["path"]),
    T("filesystem", "list_directory", "Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. Only works within allowed directories.",
      {"path": PATH}, ["path"]),
    T("filesystem", "directory_tree", "Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Only works within allowed directories.",
      {"path": PATH}, ["path"]),
    T("filesystem", "move_file", "Move or rename files and directories. Can move files between directories and rename them in a single operation. If the destination exists, the operation will fail. Both source and destination must be within allowed directories.",
      {"source": S("Source path"), "destination": S("Destination path")}, ["source", "destination"]),
    T("filesystem", "search_files", "Recursively search for files and directories matching a pattern. Searches through all subdirectories from the starting path. The search is case-insensitive and matches partial names. Only searches within allowed directories.",
      {"path": PATH, "pattern": S("Pattern to match"), "excludePatterns": A(S(), "Glob patterns to exclude", default=[])}, ["path", "pattern"]),
    T("filesystem", "get_file_info", "Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. Only works within allowed directories.",
      {"path": PATH}, ["path"]),
    T("filesystem", "list_allowed_directories", "Returns the list of directories that this server is allowed to access. Use this to understand which directories are available before trying to access files.", {}),
    # -------------------------------------------------------------------- git
    T("git", "git_status", "Shows the working tree status", {"repo_path": S()}, ["repo_path"]),
    T("git", "git_diff_unstaged", "Shows changes in the working directory that are not yet staged", {"repo_path": S()}, ["repo_path"]),
    T("git", "git_diff_staged", "Shows changes that are staged for commit", {"repo_path": S()}, ["repo_path"]),
    T("git", "git_diff", "Shows differences between branches or commits", {"repo_path": S(), "target": S()}, ["repo_path", "target"]),
    T("git", "git_commit", "Records changes to the repository", {"repo_path": S(), "message": S()}, ["repo_path", "message"]),
    T("git", "git_add", "Adds file contents to the staging area", {"repo_path": S(), "files": A(S())}, ["repo_path", "files"]),
    T("git", "git_reset", "Unstages all staged changes", {"repo_path": S()}, ["repo_path"]),
    T("git", "git_log", "Shows the commit logs", {"repo_path": S(), "max_count": N(t="integer", default=10)}, ["repo_path"]),
    T("git", "git_create_branch", "Creates a new branch from an optional base branch", {"repo_path": S(), "branch_name": S(), "base_branch": S()}, ["repo_path", "branch_name"]),
    T("git", "git_checkout", "Switches branches", {"repo_path": S(), "branch_name": S()}, ["repo_path", "branch_name"]),
    T("git", "git_show", "Shows the contents of a commit", {"repo_path": S(), "revision": S()}, ["repo_path", "revision"]),
    T("git", "git_init", "Initialize a new Git repository", {"repo_path": S()}, ["repo_path"]),
    # ----------------------------------------------------------------- github
    T("github", "create_or_update_file", "Create or update a single file in a GitHub repository",
      {"owner": S("Repository owner (username or organization)"), "repo": S("Repository name"), "path": S("Path where to create/update the file"),
       "content": S("Content of the file"), "message": S("Commit message"), "branch": S("Branch to create/update the file in"),
       "sha": S("SHA of the file being replaced (required when updating existing files)")}, ["owner", "repo", "path", "content", "message", "branch"]),
    T("github", "search_repositories", "Search for GitHub repositories",
      {"query": S("Search query (see GitHub search syntax)"), "page": N("Page number for pagination (default: 1)"), "perPage": N("Number of results per page (default: 30, max: 100)")}, ["query"]),
    T("github", "create_repository", "Create a new GitHub repository in your account",
      {"name": S("Repository name"), "description": S("Repository description"), "private": B("Whether the repository should be private"), "autoInit": B("Initialize with README.md")}, ["name"]),
    T("github", "get_file_contents", "Get the contents of a file or directory from a GitHub repository",
      {"owner": S("Repository owner (username or organization)"), "repo": S("Repository name"), "path": S("Path to the file or directory"), "branch": S("Branch to get contents from")}, ["owner", "repo", "path"]),
    T("github", "push_files", "Push multiple files to a GitHub repository in a single commit",
      {"owner": S("Repository owner"), "repo": S("Repository name"), "branch": S("Branch to push to (e.g., 'main' or 'master')"),
       "files": A(O({"path": S(), "content": S()}, ["path", "content"]), "Array of files to push"), "message": S("Commit message")}, ["owner", "repo", "branch", "files", "message"]),
    T("github", "create_issue", "Create a new issue in a GitHub repository",
      {"owner": S(), "repo": S(), "title": S(), "body": S(), "assignees": A(S()), "milestone": N(t="number"), "labels": A(S())}, ["owner", "repo", "title"]),
    T("github", "create_pull_request", "Create a new pull request in a GitHub repository",
      {"owner": S("Repository owner (username or organization)"), "repo": S("Repository name"), "title": S("Pull request title"), "body": S("Pull request body/description"),
       "head": S("The name of the branch where your changes are implemented"), "base": S("The name of the branch you want the changes pulled into"),
       "draft": B("Whether to create the pull request as a draft"), "maintainer_can_modify": B("Whether maintainers can modify the pull request")}, ["owner", "repo", "title", "head", "base"]),
    T("github", "fork_repository", "Fork a GitHub repository to your account or specified organization",
      {"owner": S("Repository owner (username or organization)"), "repo": S("Repository name"), "organization": S("Optional: organization to fork to (defaults to your personal account)")}, ["owner", "repo"]),
    T("github", "create_branch", "Create a new branch in a GitHub repository",
      {"owner": S("Repository owner (username or organization)"), "repo": S("Repository name"), "branch": S("Name for the new branch"), "from_branch": S("Optional: source branch to create from (defaults to the repository's default branch)")}, ["owner", "repo", "branch"]),
    T("github", "list_commits", "Get list of commits of a branch in a GitHub repository",
      {"owner": S(), "repo": S(), "sha": S(), "page": N(), "perPage": N()}, ["owner", "repo"]),
    T("github", "list_issues", "List issues in a GitHub repository with filtering options",
      {"owner": S(), "repo": S(), "state": S(enum=["open", "closed", "all"]), "labels": A(S()), "sort": S(enum=["created", "updated", "comments"]),
       "direction": S(enum=["asc", "desc"]), "since": S(), "page": N(), "per_page": N()}, ["owner", "repo"]),
    T("github", "update_issue", "Update an existing issue in a GitHub repository",
      {"owner": S(), "repo": S(), "issue_number": N(), "title": S(), "body": S(), "state": S(enum=["open", "closed"]), "labels": A(S()), "assignees": A(S()), "milestone": N()}, ["owner", "repo", "issue_number"]),
    T("github", "add_issue_comment", "Add a comment to an existing issue",
      {"owner": S(), "repo": S(), "issue_number": N(), "body": S()}, ["owner", "repo", "issue_number", "body"]),
    T("github", "search_code", "Search for code across GitHub repositories",
      {"q": S(), "order": S(enum=["asc", "desc"]), "page": N(minimum=1), "per_page": N(minimum=1, maximum=100)}, ["q"]),
    T("github", "search_issues", "Search for issues and pull requests across GitHub repositories",
      {"q": S(), "sort": S(enum=["comments", "reactions", "reactions-+1", "reactions--1", "reactions-smile", "reactions-thinking_face", "reactions-heart", "reactions-tada", "interactions", "created", "updated"]),
       "order": S(enum=["asc", "desc"]), "per_page": N(minimum=1, maximum=100), "page": N(minimum=1)}, ["q"]),
    T("github", "search_users", "Search for users on GitHub",
      {"q": S(), "sort": S(enum=["followers", "repositories", "joined"]), "order": S(enum=["asc", "desc"]), "per_page": N(minimum=1, maximum=100), "page": N(minimum=1)}, ["q"]),
    T("github", "get_issue", "Get details of a specific issue in a GitHub repository.", {"owner": S(), "repo": S(), "issue_number": N()}, ["owner", "repo", "issue_number"]),
    # ------------------------------------------------------------------ fetch
    T("fetch", "fetch", "Fetches a URL from the internet and optionally extracts its contents as markdown. Although originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.",
      {"url": S("URL to fetch", format="uri"), "max_length": N("Maximum number of characters to return.", t="integer", default=5000, exclusiveMinimum=0, exclusiveMaximum=1000000),
       "start_index": N("On return output starting at this character index, useful if a previous fetch was truncated and more context is required.", t="integer", default=0, minimum=0),
       "raw": B("Get the actual HTML content of the requested page, without simplification.", default=False)}, ["url"]),
    # ----------------------------------------------------------------- memory
    T("memory", "create_entities", "Create multiple new entities in the knowledge graph",
      {"entities": A(O({"name": S("The name of the entity"), "entityType": S("The type of the entity"), "observations": A(S(), "An array of observation contents associated with the entity")}, ["name", "entityType", "observations"]))}, ["entities"]),
    T("memory", "create_relations", "Create multiple new relations between entities in the knowledge graph. Relations should be in active voice",
      {"relations": A(O({"from": S("The name of the entity where the relation starts"), "to": S("The name of the entity where the relation ends"), "relationType": S("The type of the relation")}, ["from", "to", "relationType"]))}, ["relations"]),
    T("memory", "add_observations", "Add new observations to existing entities in the knowledge graph",
      {"observations": A(O({"entityName": S("The name of the entity to add the observations to"), "contents": A(S(), "An array of observation contents to add")}, ["entityName", "contents"]))}, ["observations"]),
    T("memory", "delete_entities", "Delete multiple entities and their associated relations from the knowledge graph",
      {"entityNames": A(S(), "An array of entity names to delete")}, ["entityNames"]),
    T("memory", "delete_observations", "Delete specific observations from entities in the knowledge graph",
      {"deletions": A(O({"entityName": S("The name of the entity containing the observations"), "observations": A(S(), "An array of observations to delete")}, ["entityName", "observations"]))}, ["deletions"]),
    T("memory", "delete_relations", "Delete multiple relations from the knowledge graph",
      {"relations": A(O({"from": S(), "to": S(), "relationType": S()}, ["from", "to", "relationType"]), "An array of relations to delete")}, ["relations"]),
    T("memory", "read_graph", "Read the entire knowledge graph", {}),
    T("memory", "search_nodes", "Search for nodes in the knowledge graph based on a query",
      {"query": S("The search query to match against entity names, types, and observation content")}, ["query"]),
    T("memory", "open_nodes", "Open specific nodes in the knowledge graph by their names",
      {"names": A(S(), "An array of entity names to retrieve")}, ["names"]),
    # ----------------------------------------------------------------- sqlite
    T("sqlite", "read_query", "Execute a SELECT query on the SQLite database", {"query": S("SELECT SQL query to execute")}, ["query"]),
    T("sqlite", "write_query", "Execute an INSERT, UPDATE, or DELETE query on the SQLite database", {"query": S("SQL query to execute")}, ["query"]),
    T("sqlite", "create_table", "Create a new table in the SQLite database", {"query": S("CREATE TABLE SQL statement")}, ["query"]),
    T("sqlite", "list_tables", "List all tables in the SQLite database", {}),
    T("sqlite", "describe_table", "Get the schema information for a specific table", {"table_name": S("Name of the table to describe")}, ["table_name"]),
    T("sqlite", "append_insight", "Add a business insight to the memo", {"insight": S("Business insight discovered from data analysis")}, ["insight"]),
    # ------------------------------------------------------------------- time
    T("time", "get_current_time", "Get current time in a specific timezones",
      {"timezone": S("IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'UTC' as local timezone if no timezone provided by the user.")}, ["timezone"]),
    T("time", "convert_time", "Convert time between timezones",
      {"source_timezone": S("Source IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'UTC' as local timezone if no source timezone provided by the user."),
       "time": S("Time to convert in 24-hour format (HH:MM)"),
       "target_timezone": S("Target IANA timezone name (e.g., 'Asia/Tokyo', 'America/San_Francisco'). Use 'UTC' as local timezone if no target timezone provided by the user.")},
      ["source_timezone", "time", "target_timezone"]),
    # ----------------------------------------------------------- brave-search
    T("brave-search", "brave_web_search", "Performs a web search using the Brave Search API, ideal for general queries, news, articles, and online content. Use this for broad information gathering, recent events, or when you need diverse web sources. Supports pagination, content filtering, and freshness controls. Maximum 20 results per request, with offset for pagination.",
      {"query": S("Search query (max 400 chars, 50 words)"), "count": N("Number of results (1-20, default 10)", default=10), "offset": N("Pagination offset (max 9, default 0)", default=0)}, ["query"]),
    T("brave-search", "brave_local_search", "Searches for local businesses and places using Brave's Local Search API. Best for queries related to physical locations, businesses, restaurants, services, etc. Returns detailed information including business names and addresses, ratings and review counts, phone numbers and opening hours. Use this when the query implies 'near me' or mentions specific locations. Automatically falls back to web search if no local results are found.",
      {"query": S("Local search query (e.g. 'pizza near Central Park')"), "count": N("Number of results (1-20, default 5)", default=5)}, ["query"]),
    # -------------------------------------------------------------- puppeteer
    T("puppeteer", "puppeteer_navigate", "Navigate to a URL", {"url": S("URL to navigate to"), "launchOptions": O({}, desc="PuppeteerJS LaunchOptions. Default null. If changed and not null, browser restarts."), "allowDangerous": B("Allow dangerous LaunchOptions that reduce security.")}, ["url"]),
    T("puppeteer", "puppeteer_screenshot", "Take a screenshot of the current page or a specific element",
      {"name": S("Name for the screenshot"), "selector": S("CSS selector for element to screenshot"), "width": N("Width in pixels (default: 800)"), "height": N("Height in pixels (default: 600)")}, ["name"]),
    T("puppeteer", "puppeteer_click", "Click an element on the page", {"selector": S("CSS selector for element to click")}, ["selector"]),
    T("puppeteer", "puppeteer_fill", "Fill out an input field", {"selector": S("CSS selector for input field"), "value": S("Value to fill")}, ["selector", "value"]),
    T("puppeteer", "puppeteer_select", "Select an element on the page with Select tag", {"selector": S("CSS selector for element to select"), "value": S("Value to select")}, ["selector", "value"]),
    T("puppeteer", "puppeteer_hover", "Hover an element on the page", {"selector": S("CSS selector for element to hover")}, ["selector"]),
    T("puppeteer", "puppeteer_evaluate", "Execute JavaScript in the browser console", {"script": S("JavaScript code to execute")}, ["script"]),
    # ------------------------------------------------------------------ slack
    T("slack", "slack_list_channels", "List public or pre-defined channels in the workspace with pagination",
      {"limit": N("Maximum number of channels to return (default 100, max 200)", default=100), "cursor": S("Pagination cursor for next page of results")}),
    T("slack", "slack_post_message", "Post a new message to a Slack channel",
      {"channel_id": S("The ID of the channel to post to"), "text": S("The message text to post")}, ["channel_id", "text"]),
    T("slack", "slack_reply_to_thread", "Reply to a specific message thread in Slack",
      {"channel_id": S("The ID of the channel containing the thread"), "thread_ts": S("The timestamp of the parent message in the format '1234567890.123456'. Timestamps in the format without the period can be converted by adding the period such that 6 numbers come after it."),
       "text": S("The reply text")}, ["channel_id", "thread_ts", "text"]),
    T("slack", "slack_add_reaction", "Add a reaction emoji to a message",
      {"channel_id": S("The ID of the channel containing the message"), "timestamp": S("The timestamp of the message to react to"), "reaction": S("The name of the emoji reaction (without ::)")}, ["channel_id", "timestamp", "reaction"]),
    T("slack", "slack_get_channel_history", "Get recent messages from a channel",
      {"channel_id": S("The ID of the channel"), "limit": N("Number of messages to retrieve (default 10)", default=10)}, ["channel_id"]),
    T("slack", "slack_get_thread_replies", "Get all replies in a message thread",
      {"channel_id": S("The ID of the channel containing the thread"), "thread_ts": S("The timestamp of the parent message in the format '1234567890.123456'. Timestamps in the format without the period can be converted by adding the period such that 6 numbers come after it.")}, ["channel_id", "thread_ts"]),
    T("slack", "slack_get_users", "Get a list of all users in the workspace with their basic profile information",
      {"cursor": S("Pagination cursor for next page of results"), "limit": N("Maximum number of users to return (default 100, max 200)", default=100)}),
    T("slack", "slack_get_user_profile", "Get detailed profile information for a specific user", {"user_id": S("The ID of the user")}, ["user_id"]),
    # ------------------------------------------------------------ google-maps
    T("google-maps", "maps_geocode", "Convert an address into geographic coordinates", {"address": S("The address to geocode")}, ["address"]),
    T("google-maps", "maps_reverse_geocode", "Convert coordinates into an address", {"latitude": N("Latitude coordinate"), "longitude": N("Longitude coordinate")}, ["latitude", "longitude"]),
    T("google-maps", "maps_search_places", "Search for places using Google Places API",
      {"query": S("Search query"), "location": O({"latitude": N(), "longitude": N()}, desc="Optional center point for the search"), "radius": N("Search radius in meters (max 50000)")}, ["query"]),
    T("google-maps", "maps_place_details", "Get detailed information about a specific place", {"place_id": S("The place ID to get details for")}, ["place_id"]),
    T("google-maps", "maps_distance_matrix", "Calculate travel distance and time for multiple origins and destinations",
      {"origins": A(S(), "Array of origin addresses or coordinates"), "destinations": A(S(), "Array of destination addresses or coordinates"),
       "mode": S("Travel mode (driving, walking, bicycling, transit)", enum=["driving", "walking", "bicycling", "transit"])}, ["origins", "destinations"]),
    T("google-maps", "maps_elevation", "Get elevation data for locations on the earth",
      {"locations": A(O({"latitude": N(), "longitude": N()}, ["latitude", "longitude"]), "Array of locations to get elevation for")}, ["locations"]),
    T("google-maps", "maps_directions", "Get directions between two points",
      {"origin": S("Starting point address or coordinates"), "destination": S("Ending point address or coordinates"),
       "mode": S("Travel mode (driving, walking, bicycling, transit)", enum=["driving", "walking", "bicycling", "transit"])}, ["origin", "destination"]),
    # ------------------------------------------------------------- everything
    T("everything", "echo", "Echoes back the input", {"message": S("Message to echo")}, ["message"]),
    T("everything", "add", "Adds two numbers", {"a": N("First number"), "b": N("Second number")}, ["a", "b"]),
    T("everything", "longRunningOperation", "Demonstrates a long running operation with progress updates",
      {"duration": N("Duration of the operation in seconds", default=10), "steps": N("Number of steps in the operation", default=5)}),
    T("everything", "sampleLLM", "Samples from an LLM using MCP's sampling feature",
      {"prompt": S("The prompt to send to the LLM"), "maxTokens": N("Maximum number of tokens to generate", default=100)}, ["prompt"]),
    T("everything", "getTinyImage", "Returns the MCP_TINY_IMAGE", {}),
    T("everything", "printEnv", "Prints all environment variables, helpful for debugging MCP server configuration", {}),
    T("everything", "annotatedMessage", "Demonstrates how annotations can be used to provide metadata about content",
      {"messageType": S("Type of message to demonstrate different annotation patterns", enum=["error", "success", "debug"]), "includeImage": B("Whether to include an example image", default=False)}, ["messageType"]),
    T("everything", "getResourceReference", "Returns a resource reference that can be used by MCP clients",
      {"resourceId": N("ID of the resource to reference (1-100)", t="number", minimum=1, maximum=100)}, ["resourceId"]),
    # ---------------------------------------------------- sequential-thinking
    T("sequential-thinking", "sequentialthinking", "A detailed tool for dynamic and reflective problem-solving through thoughts. This tool helps analyze problems through a flexible thinking process that can adapt and evolve. Each thought can build on, question, or revise previous insights as understanding deepens.",
      {"thought": S("Your current thinking step"), "nextThoughtNeeded": B("Whether another thought step is needed"),
       "thoughtNumber": N("Current thought number", t="integer", minimum=1), "totalThoughts": N("Estimated total thoughts needed", t="integer", minimum=1),
       "isRevision": B("Whether this revises previous thinking"), "revisesThought": N("Which thought is being reconsidered", t="integer", minimum=1),
       "branchFromThought": N("Branching point thought number", t="integer", minimum=1), "branchId": S("Branch identifier"),
       "needsMoreThoughts": B("If more thoughts are needed")}, ["thought", "nextThoughtNeeded", "thoughtNumber", "totalThoughts"]),
    # --------------------------------------------------------------- postgres
    T("postgres", "query", "Run a read-only SQL query", {"sql": S()}, ["sql"]),
    # ------------------------------------------------------------- everart / aws-kb
    T("everart", "generate_image", "Generate images using EverArt Models and returns a clickable link to view the generated image. The tool will return a URL that can be clicked to view the image in a browser. Available models:\n- 5000:FLUX1.1: Standard quality\n- 9000:FLUX1.1-ultra: Ultra high quality\n- 6000:SD3.5: Stable Diffusion 3.5\n- 7000:Recraft-Real: Photorealistic style\n- 8000:Recraft-Vector: Vector art style\n\nThe response will contain a direct link to view the generated image.",
      {"prompt": S("Text description of desired image"), "model": S("Model ID (5000:FLUX1.1, 9000:FLUX1.1-ultra, 6000:SD3.5, 7000:Recraft-Real, 8000:Recraft-Vector)", default="5000"),
       "image_count": N("Number of images to generate", default=1)}, ["prompt"]),
    T("aws-kb-retrieval", "retrieve_from_aws_kb", "Performs retrieval from the AWS Knowledge Base using the provided query and Knowledge Base ID.",
      {"query": S("The query to perform retrieval on"), "knowledgeBaseId": S("The ID of the AWS Knowledge Base"), "n": N("Number of results to retrieve", default=3)}, ["query", "knowledgeBaseId"]),
    T("gdrive", "search", "Search for files in Google Drive", {"query": S("Search query")}, ["query"]),
    T("redis", "set", "Set a Redis key-value pair with optional expiration",
      {"key": S("Redis key"), "value": S("Value to store"), "expireSeconds": N("Optional expiration time in seconds")}, ["key", "value"]),
    T("redis", "get", "Get value by key from Redis", {"key": S("Redis key to retrieve")}, ["key"]),
    T("redis", "delete", "Delete one or more keys from Redis", {"key": {"oneOf": [S(), A(S())], "description": "Key or array of keys to delete"}}, ["key"]),
    T("redis", "list", "List Redis keys matching a pattern", {"pattern": S("Pattern to match keys (default: *)", default="*")}),
]


def build() -> list[dict]:
    out = []
    for t in TOOLS:
        out.append({**t, "provenance": "REAL_TOOL_DEFINITION",
                    "provenance_note": "Transcribed from public MCP reference-server documentation (modelcontextprotocol/servers)."})
    return out


def main() -> None:
    cfg = load_config()
    tools = build()
    write_json(tools, resolve(cfg["data"]["mcp_reference_file"]))
    print(f"wrote {len(tools)} MCP reference tools")


if __name__ == "__main__":
    main()
