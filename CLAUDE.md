# Portfolio v2

## Context
- Owner: Chaitanya, AI Engineer at Qlik
- Live site: chaitanya-19.github.io (Astro + AstroPaper template currently)
- Goal: rebuild combining blog + portfolio, modernize design, fix template placeholders
- Working branch: v2-rebuild; main stays live until cutover

## Stack
- Astro + MDX + TypeScript
- Styling: TBD in Phase 1 (Tailwind vs vanilla CSS — decide together)
- Deploy: GitHub Actions → GitHub Pages

## Existing content to migrate
- Blog: "Counting Every Parameter in GPT-2", GQA deep dive
- About page: thin, needs full rewrite
- Footer: still has AstroPaper template placeholders — purge all of these

## Conventions
- TypeScript, not JS
- Components in src/components/, content in src/content/
- `pnpm build` must pass before any commit
- One concern per PR
- Package manager: pnpm (pinned via `packageManager` field in package.json — do not use npm or yarn)

## Voice / prose style (matters for any content you draft)
- Direct, technical, mechanistic
- No "in today's fast-paced world", no AI cadence
- Concrete numbers and specifics over abstractions
- Short sentences over long ones

## Do not touch
- .github/workflows/ until Phase 3
- main branch directly — PRs only after Phase 3 lands

## One-time setup
```
corepack enable   # enforces packageManager pin; run once per machine
pnpm install
```

## Build commands
| Command | Purpose |
|---|---|
| `pnpm dev` | local dev server |
| `pnpm build` | type-check + build + pagefind index |
| `pnpm preview` | serve built output |
| `pnpm format` | prettier write |
| `pnpm lint` | eslint |