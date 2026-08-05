# Pages redirect

Two-file static site whose only job is to bounce
`https://liujonathan24.github.io/NetHack-engine/` to the console's real home:

<https://liujonathan24.github.io/NetHack-console/>

## Why this exists

This repo used to publish the browser console from `deploy/wasm-web/`. That
directory moved to [`NetHack-console`](https://github.com/liujonathan24/NetHack-console)
when the monolith was split (#44), and the console repo now owns the deployment.
This repo's Pages URL predates the split and is linked from the README, the
site's own nav, and outside pages, so it stays alive as a redirect rather than
serving a stale copy or 404ing.

`index.html` covers the root; `404.html` covers every other path, which is how a
deep link like `/NetHack-engine/play.html` reaches `/NetHack-console/play.html`.
Both carry a `<meta http-equiv="refresh">` fallback for when scripting is off.

## If you would rather not keep a redirect

Delete this directory and `.github/workflows/deploy-pages-redirect.yml`, then
turn Pages off for the repo (Settings → Pages → Source: None, or
`gh api -X DELETE repos/liujonathan24/NetHack-engine/pages`). The old URL will
then 404 instead of forwarding.
