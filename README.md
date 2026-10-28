# Rethinking Asymmetry in Self-Supervised 3D Geometry Learning — project page

Static site: `index.html` + `static/`. No build step and no external libraries.

- `static/img/`: figures and teaser; `*_thumbs/` are input-view thumbnails for the interactive viewers
- `static/video/`: reconstruction clips (carousel)
- `static/data/`: camera trajectories (`trajectories.js`, `dyn_trajectories.js`) and point clouds for the
  interactive reconstruction viewer (`recon/*.js`, loaded on demand)
- `static/js/`: trajectory viewer (canvas) and point-cloud viewer (WebGL)

Data ships as `.js` files, so the page also works when opened directly from disk.

This is the `gh-pages` branch of https://github.com/Gynjn/rethinking-asymmetry. Hosting with GitHub Pages: Settings → Pages → Deploy from a branch → `gh-pages` / root.
