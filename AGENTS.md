# Agent instructions

## Work items come from Runqueue (project TRA)

- Connect the Runqueue MCP server: https://app.runqueue.ai/mcp
- Start with `claim_next` (project TRA), or `claim_item TRA-<n>` for a specific item. To file work you are about to do yourself, use `create_item` with `take: true`.
- Work on the branch the answer gives you (`pr0/TRA-<n>-<title>`) and put `pr0: TRA-<n>` in the PR description. Opening the PR moves the item to review; merging moves it to done. Pushes keep the item yours.
- Stuck? `ask` the project's lead (PM), designer or QA role (`project_info` lists the roles), or `release` the item with the reason.
- Changes under `**/auth/**`, `**/payments/**`, `**/migrations/**`, `infra/**`, `.github/**` need a person's review; say so in the PR.
- Before opening a PR, check the item's acceptance criteria are met.