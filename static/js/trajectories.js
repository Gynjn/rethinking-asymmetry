// Interactive 3D camera-trajectory viewer, styled after the supplementary video (dark 3D axis box,
// white ground truth, colored predictions; cameras appear in temporal order, then the view orbits).
// No dependencies: plain canvas 2D with a small perspective projector.
// Data: window.TRAJ_DATA (static/data/trajectories.js) from tools/export_trajectories.py — per scene,
// GT and Sim(3)-aligned predictions (t: centres, R: camera-to-world rotations, OpenCV convention),
// normalised to unit GT extent, plus per-scene AUC@30 / RPA@5 / APE of each method.
function initTrajViewer(P, DATA) {   // P: id prefix of the viewer's controls (e.g. 'traj', 'dyn')
  DATA = DATA || [];
  var C = { bg: '#0e0f14', pane: '#161821', grid: '#2a2d3a', edge: '#343849', gt: '#f2f2f5', link: 'rgba(200,202,214,0.35)' };
  var COLOR = { 'SPFSplat': '#ff6b5e', 'E-RayZer': '#ff6b5e', 'NAS3R': '#5fd38d', 'Ours': '#a78bfa',
                'DL3DV': '#ff6b5e', 'DL3DV + SpatialVID': '#a78bfa' };
  var elTrained = document.getElementById(P + 'Trained');
  var elScene = document.getElementById(P + 'Scene');
  var elPanels = document.getElementById(P + 'Panels');
  var elThumbs = document.getElementById(P + 'Thumbs');
  if (!elPanels) return;
  if (!DATA.length) {
    elPanels.innerHTML = '<p class="traj-msg">The trajectory data could not be loaded.</p>';
    return;
  }
  var reduce = (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) || /still/.test(location.hash);
  var DEFAULT = { az: -0.6, el: 0.5, zoom: 1, panX: 0, panY: 0 };
  var view = Object.assign({}, DEFAULT);
  // which side of each axis is "back" in the default view (z2 grows toward the viewer)
  var FIXED_PANES = [1, 0, 2].map(function (axis) {
    var ca = Math.cos(DEFAULT.az), sa = Math.sin(DEFAULT.az), ce = Math.cos(DEFAULT.el), se = Math.sin(DEFAULT.el);
    var v = [0, 0, 0]; v[axis] = 1;
    var z1 = -sa * v[0] + ca * v[2], z2 = se * v[1] + ce * z1;   // depth of the +1 pane centre
    return [axis, z2 > 0 ? -1 : 1];
  });
  var panels = [], reveal = 1, orbit = !reduce, t0 = 0, raf = 0;
  var T_IN = 2500;   // ms for the 10 cameras to appear

  function scenesFor(trained) { return DATA.filter(function (d) { return d.trained === trained; }); }
  function W(p) { return [p[0], -p[1], p[2]]; }   // OpenCV (y down) -> display (y up)

  function frustum(R, t, size) {
    var h = Math.tan(25 * Math.PI / 180) * size, w = h * 4 / 3;
    var local = [[0, 0, 0], [w, h, size], [w, -h, size], [-w, -h, size], [-w, h, size]];
    var p = local.map(function (q) {
      return W([0, 1, 2].map(function (i) { return R[i][0] * q[0] + R[i][1] * q[1] + R[i][2] * q[2] + t[i]; }));
    });
    return { edges: [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]].map(function (e) { return [p[e[0]], p[e[1]]]; }),
             face: [p[1], p[2], p[3], p[4]] };
  }

  function prepare(sc) {
    var size = 0.075;
    var all = sc.gt.t.slice();
    sc.methods.forEach(function (m) { all = all.concat(m.t); });
    all = all.map(W);
    var lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    all.forEach(function (p) { for (var i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); } });
    var c = lo.map(function (l, i) { return (l + hi[i]) / 2; });
    var half = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2 * 1.12 + size;
    function geo(cam) { return { fr: cam.t.map(function (t, i) { return frustum(cam.R[i], t, size); }), path: cam.t.map(W) }; }
    return { c: c, half: half, gt: geo(sc.gt),
      methods: sc.methods.map(function (m) {
        var q = geo(m);
        q.links = sc.gt.t.map(function (g, i) { return [W(g), W(m.t[i])]; });
        return q;
      }) };
  }

  function projector(w, h, P) {
    var ca = Math.cos(view.az), sa = Math.sin(view.az), ce = Math.cos(view.el), se = Math.sin(view.el);
    var s = Math.min(w, h) * 0.31 * view.zoom / P.half, dist = P.half * 5;
    return function (p) {
      var x = p[0] - P.c[0], y = p[1] - P.c[1], z = p[2] - P.c[2];
      var x1 = ca * x + sa * z, z1 = -sa * x + ca * z;
      var y2 = ce * y - se * z1, z2 = se * y + ce * z1;
      var k = dist / (dist - z2);                                // z2 points toward the viewer
      return [w / 2 + view.panX + x1 * s * k, h * 0.54 + view.panY - y2 * s * k, z2];
    };
  }

  function drawBox(ctx, pr, P) {
    var c = P.c, a = P.half, n = 8;
    function pt(u, v, axis, level) {   // point on a pane: axis = fixed coordinate index, level = -1|1
      var q = [0, 0, 0], free = [0, 1, 2].filter(function (i) { return i !== axis; });
      q[axis] = c[axis] + level * a; q[free[0]] = c[free[0]] + u * a; q[free[1]] = c[free[1]] + v * a;
      return pr(q);
    }
    // open box: the panes are the ones behind the scene in the DEFAULT view (floor + two back walls)
    // and stay fixed while the user rotates
    var panes = FIXED_PANES;
    panes.forEach(function (pn) {
      var corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(function (uv) { return pt(uv[0], uv[1], pn[0], pn[1]); });
      ctx.fillStyle = C.pane; ctx.strokeStyle = C.edge; ctx.lineWidth = 1;
      ctx.beginPath(); corners.forEach(function (q, i) { if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = C.grid; ctx.beginPath();
      for (var k = 1; k < n; k++) {
        var u = -1 + 2 * k / n, p1 = pt(u, -1, pn[0], pn[1]), p2 = pt(u, 1, pn[0], pn[1]);
        var p3 = pt(-1, u, pn[0], pn[1]), p4 = pt(1, u, pn[0], pn[1]);
        ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.moveTo(p3[0], p3[1]); ctx.lineTo(p4[0], p4[1]);
      }
      ctx.stroke();
    });
  }

  function drawCams(ctx, pr, g, color, count, width) {
    var shown = Math.ceil(count);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = color; ctx.lineWidth = width + 0.8; ctx.beginPath();
    g.path.slice(0, shown).forEach(function (p, i) { var a = pr(p); if (i) ctx.lineTo(a[0], a[1]); else ctx.moveTo(a[0], a[1]); });
    ctx.stroke();
    for (var i = 0; i < shown; i++) {
      var f = g.fr[i], alpha = Math.min(1, count - i);
      ctx.globalAlpha = 0.18 * alpha; ctx.fillStyle = color; ctx.beginPath();
      f.face.forEach(function (p, j) { var a = pr(p); if (j) ctx.lineTo(a[0], a[1]); else ctx.moveTo(a[0], a[1]); });
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      f.edges.forEach(function (s) { var a = pr(s[0]), b = pr(s[1]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); });
      ctx.stroke();
      var c0 = pr(g.path[i]); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(c0[0], c0[1], 2.2, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function draw(panel) {
    var cv = panel.canvas, ctx = cv.getContext('2d'), dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
    var P = panel.prep, pr = projector(w, h, P), m = P.methods[panel.idx], count = reveal * m.path.length;
    drawBox(ctx, pr, P);
    ctx.strokeStyle = C.link; ctx.lineWidth = 1; ctx.beginPath();
    m.links.slice(0, Math.ceil(count)).forEach(function (s) { var a = pr(s[0]), b = pr(s[1]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); });
    ctx.stroke();
    drawCams(ctx, pr, P.gt, C.gt, count, 1.3);
    drawCams(ctx, pr, m, panel.color, count, 1.7);
  }

  function frame(now) {
    raf = 0;
    if (!t0) t0 = now;
    reveal = reduce ? 1 : Math.min(1, (now - t0) / T_IN);
    if (orbit && reveal >= 1) view.az += 0.0035;
    panels.forEach(draw);
    if (reveal < 1 || orbit) raf = requestAnimationFrame(frame);
  }
  function kick() { if (!raf) raf = requestAnimationFrame(frame); }
  function stopOrbit() { orbit = false; }
  function redraw() { panels.forEach(draw); }

  function attach(cv) {
    var drag = null;
    cv.addEventListener('pointerdown', function (e) {
      stopOrbit(); reveal = 1;
      drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, id: e.pointerId };
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', function (e) {
      if (!drag || drag.id !== e.pointerId) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) { view.panX += dx; view.panY += dy; }
      else { view.az += dx * 0.01; view.el = Math.max(-1.45, Math.min(1.45, view.el + dy * 0.01)); }
      redraw();
    });
    function end() { drag = null; }
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    cv.addEventListener('wheel', function (e) {
      e.preventDefault(); stopOrbit(); reveal = 1;
      view.zoom = Math.max(0.3, Math.min(8, view.zoom * Math.exp(-e.deltaY * 0.0015)));
      redraw();
    }, { passive: false });
    cv.addEventListener('dblclick', restart);
  }

  function restart() {
    view = Object.assign({}, DEFAULT); orbit = !reduce; t0 = 0; reveal = reduce ? 1 : 0;
    kick(); redraw();
  }

  function render() {
    var list = scenesFor(elTrained.value);
    var scene = list[+elScene.value] || list[0];
    if (!scene) return;
    elThumbs.innerHTML = scene.thumbs.map(function (src, i) {
      return '<img src="' + src + '" alt="Input view ' + (i + 1) + ' of scene ' + scene.scene + '">';
    }).join('');
    var prep = prepare(scene);
    var bestAuc = Math.max.apply(null, scene.methods.map(function (m) { return m.auc30; }));
    var bestApe = Math.min.apply(null, scene.methods.map(function (m) { return m.ape; }));
    elPanels.innerHTML = '';
    elPanels.style.setProperty('--n', scene.methods.length);
    panels = scene.methods.map(function (m, i) {
      var col = COLOR[m.name] || m.color;
      var wrap = document.createElement('div');
      wrap.className = 'traj-panel';
      wrap.innerHTML = '<div class="traj-head" style="color:' + col + '"><div class="nm">' + m.name + '</div>' +
        '<div class="mt"><span' + (m.auc30 === bestAuc ? ' class="best"' : '') + '>AUC@30 ' + m.auc30.toFixed(1) + '</span>' +
        '<span>RPA@5 ' + m.rpa5.toFixed(1) + '</span>' +
        '<span' + (m.ape === bestApe ? ' class="best"' : '') + '>APE ' + m.ape.toFixed(3) + '</span></div></div>';
      var cv = document.createElement('canvas');
      cv.className = 'traj-plot';
      cv.setAttribute('role', 'img');
      cv.setAttribute('aria-label', m.name + ' camera trajectory against ground truth: AUC@30 ' + m.auc30.toFixed(1) +
        ', RPA@5 ' + m.rpa5.toFixed(1) + ', APE ' + m.ape.toFixed(3));
      wrap.appendChild(cv);
      elPanels.appendChild(wrap);
      attach(cv);
      return { canvas: cv, prep: prep, idx: i, color: col };
    });
    restart();
  }

  function fillScenes() {
    elScene.innerHTML = scenesFor(elTrained.value).map(function (d, i) {
      return '<option value="' + i + '">' + d.group + ' · ' + d.scene.slice(0, 16) + '</option>';
    }).join('');
  }
  function step(k) {
    var n = elScene.options.length;
    elScene.value = String((+elScene.value + k + n) % n);
    render();
  }

  elTrained.addEventListener('change', function () { fillScenes(); render(); });
  elScene.addEventListener('change', render);
  document.getElementById(P + 'Prev').addEventListener('click', function () { step(-1); });
  document.getElementById(P + 'Next').addEventListener('click', function () { step(1); });
  document.getElementById(P + 'Reset').addEventListener('click', restart);
  window.addEventListener('resize', redraw);
  if (P === 'traj' && /dl3dv/i.test(location.hash)) elTrained.value = 'DL3DV-trained';
  fillScenes();
  render();
}

initTrajViewer('traj', window.TRAJ_DATA);
initTrajViewer('dyn', window.DYN_TRAJ_DATA);
