// Interactive point-cloud viewer for the reconstruction carousel (plain WebGL, no libraries).
// Each slide with data-recon="<slug>" gets an "Explore in 3D" button; the scene's data
// (static/data/recon/<slug>.js) is loaded on first click via a <script> tag, so it also works from file://.
// Same content as the clip: the model's lifted points, the predicted input frusta and their path.
(function () {
  var ACCENT = [139 / 255, 92 / 255, 246 / 255], PATH = [200 / 255, 185 / 255, 1];
  var FOCAL = 1.25;                       // focal / (half width), as in the clips

  function loadScene(slug, cb) {
    if (window.RECON_DATA && window.RECON_DATA[slug]) return cb(window.RECON_DATA[slug]);
    var s = document.createElement('script'); s.src = 'static/data/recon/' + slug + '.js';
    s.onload = function () { cb(window.RECON_DATA && window.RECON_DATA[slug]); };
    s.onerror = function () { cb(null); };
    document.head.appendChild(s);
  }
  function b64(str, Type) {
    var bin = atob(str), n = bin.length, u8 = new Uint8Array(n);
    for (var i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
    return new Type(u8.buffer);
  }
  var sub = function (a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; };
  var add = function (a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; };
  var mul = function (a, s) { return [a[0] * s, a[1] * s, a[2] * s]; };
  var dot = function (a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  var cross = function (a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; };
  var norm = function (a) { var l = Math.sqrt(dot(a, a)) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  function compile(gl, vs, fs) {
    function sh(t, src) { var s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s); return s; }
    var p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p); return p;
  }
  // camera: OpenCV convention (x right, y down, z forward); rows of R are the camera axes in world
  var CAM = 'uniform vec3 uEye, uX, uY, uZ; uniform vec2 uF; uniform vec2 uNF;\n' +
    'vec4 proj(vec3 P) { vec3 d = P - uEye; vec3 c = vec3(dot(d, uX), dot(d, uY), dot(d, uZ));\n' +
    '  float A = (uNF.y + uNF.x) / (uNF.y - uNF.x), B = 2.0 * uNF.x * uNF.y / (uNF.y - uNF.x);\n' +
    '  return vec4(uF.x * c.x, -uF.y * c.y, A * c.z - B, c.z); }\n';
  var PTS_VS = 'attribute vec3 aQ; attribute vec3 aC; uniform vec3 uLo, uSc; uniform float uSize; varying vec3 vC;\n' + CAM +
    'void main() { vec4 p = proj(uLo + aQ * uSc); gl_Position = p; gl_PointSize = clamp(uSize / max(p.w, 1e-4), 1.0, 9.0); vC = aC; }';
  var PTS_FS = 'precision mediump float; varying vec3 vC; void main() { gl_FragColor = vec4(vC, 1.0); }';
  var LIN_VS = 'attribute vec3 aP; ' + CAM + 'void main() { gl_Position = proj(aP); }';
  var LIN_FS = 'precision mediump float; uniform vec3 uCol; void main() { gl_FragColor = vec4(uCol, 1.0); }';

  function Viewer(box, D, label) {
    var canvas = document.createElement('canvas'); canvas.className = 'r3-canvas';
    var strip = document.createElement('div'); strip.className = 'r3-strip';
    D.thumbs.forEach(function (src) { var im = new Image(); im.src = src; im.alt = ''; strip.appendChild(im); });
    var lab = document.createElement('div'); lab.className = 'r3-label'; lab.textContent = label;
    var hint = document.createElement('div'); hint.className = 'r3-hint';
    hint.textContent = 'drag: rotate · scroll / pinch: zoom · right-drag / shift-drag: pan';
    box.appendChild(strip); box.appendChild(canvas); box.appendChild(lab); box.appendChild(hint);
    var gl = canvas.getContext('webgl', { antialias: true }) || canvas.getContext('experimental-webgl');
    if (!gl) { hint.textContent = 'WebGL is not available in this browser.'; return { reset: function () {}, draw: function () {} }; }

    var pp = compile(gl, PTS_VS, PTS_FS), lp = compile(gl, LIN_VS, LIN_FS);
    var qb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qb); gl.bufferData(gl.ARRAY_BUFFER, b64(D.pos, Uint16Array), gl.STATIC_DRAW);
    var cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, b64(D.col, Uint8Array), gl.STATIC_DRAW);
    // frusta (accent) and camera path (light), built in world space from the predicted c2w
    var fr = [], pa = [], s = D.extent * 0.09, hh = Math.tan(25 * Math.PI / 180) * s;
    D.cams.forEach(function (T, i) {
      var R = function (v) { return [T[0][0] * v[0] + T[0][1] * v[1] + T[0][2] * v[2] + T[0][3],
                                      T[1][0] * v[0] + T[1][1] * v[1] + T[1][2] * v[2] + T[1][3],
                                      T[2][0] * v[0] + T[2][1] * v[1] + T[2][2] * v[2] + T[2][3]]; };
      var q = [[0, 0, 0], [hh, hh, s], [hh, -hh, s], [-hh, -hh, s], [-hh, hh, s]].map(R);
      [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]].forEach(function (e) { fr.push.apply(fr, q[e[0]].concat(q[e[1]])); });
      if (i > 0) pa.push.apply(pa, R.call(null, [0, 0, 0]).concat([D.cams[i - 1][0][3], D.cams[i - 1][1][3], D.cams[i - 1][2][3]]));
    });
    var fb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, fb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(fr), gl.STATIC_DRAW);
    var pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pa), gl.STATIC_DRAW);

    var fwd = D.fwd, up = D.up, side = cross(fwd, up);
    var st = {}, W = 1, H = 1, dpr = 1, raf = 0, idle = true, t0 = performance.now();
    function fitDist() { return D.radius * FOCAL * (W / H) * 1.05; }   // cloud fills the view height
    function reset() { st = { az: 0, el: 14 * Math.PI / 180, dist: fitDist(), target: D.center.slice() }; idle = true; t0 = performance.now(); draw(); }
    function eyeFrame() {
      var ce = Math.cos(st.el);
      var back = add(add(mul(fwd, -Math.cos(st.az) * ce), mul(side, Math.sin(st.az) * ce)), mul(up, Math.sin(st.el)));
      var eye = add(st.target, mul(back, st.dist));
      var z = norm(sub(st.target, eye)), x = norm(cross(z, up)), y = cross(z, x);
      return { eye: eye, x: x, y: y, z: z };
    }
    function setCam(p, F) {
      gl.uniform3fv(gl.getUniformLocation(p, 'uEye'), F.eye); gl.uniform3fv(gl.getUniformLocation(p, 'uX'), F.x);
      gl.uniform3fv(gl.getUniformLocation(p, 'uY'), F.y); gl.uniform3fv(gl.getUniformLocation(p, 'uZ'), F.z);
      gl.uniform2f(gl.getUniformLocation(p, 'uF'), FOCAL, FOCAL * W / H);
      gl.uniform2f(gl.getUniformLocation(p, 'uNF'), st.dist * 0.01, st.dist * 20 + D.radius * 10);
    }
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = Math.max(1, canvas.clientWidth); H = Math.max(1, canvas.clientHeight);
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    }
    function draw() {
      if (raf) return;
      raf = requestAnimationFrame(function () {
        raf = 0;
        if (canvas.clientWidth !== W || canvas.clientHeight !== H) resize();
        if (idle) {   // gentle sweep like the clip until the first interaction
          var u = (performance.now() - t0) / 8000;
          st.az = 38 * Math.PI / 180 * Math.sin(2 * Math.PI * u) * 0.6;
          st.el = (14 + 8 * Math.sin(Math.PI * u) * Math.sin(Math.PI * u)) * Math.PI / 180;
        }
        var F = eyeFrame();
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(14 / 255, 15 / 255, 20 / 255, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.useProgram(pp); setCam(pp, F);
        gl.uniform3fv(gl.getUniformLocation(pp, 'uLo'), D.lo); gl.uniform3fv(gl.getUniformLocation(pp, 'uSc'), D.sc);
        // ~1.8 px at the default distance on a 1280-px-wide view, growing as the camera moves in
        gl.uniform1f(gl.getUniformLocation(pp, 'uSize'), 1.8 * (canvas.width / 1280 + 0.5) * fitDist());
        var aQ = gl.getAttribLocation(pp, 'aQ'), aC = gl.getAttribLocation(pp, 'aC');
        gl.bindBuffer(gl.ARRAY_BUFFER, qb); gl.enableVertexAttribArray(aQ); gl.vertexAttribPointer(aQ, 3, gl.UNSIGNED_SHORT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.enableVertexAttribArray(aC); gl.vertexAttribPointer(aC, 3, gl.UNSIGNED_BYTE, true, 0, 0);
        gl.drawArrays(gl.POINTS, 0, D.n);
        gl.disableVertexAttribArray(aC);
        gl.useProgram(lp); setCam(lp, F);
        var aP = gl.getAttribLocation(lp, 'aP'); gl.enableVertexAttribArray(aP);
        gl.bindBuffer(gl.ARRAY_BUFFER, fb); gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0);
        gl.uniform3fv(gl.getUniformLocation(lp, 'uCol'), ACCENT); gl.drawArrays(gl.LINES, 0, fr.length / 3);
        gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0);
        gl.uniform3fv(gl.getUniformLocation(lp, 'uCol'), PATH); gl.drawArrays(gl.LINES, 0, pa.length / 3);
        if (idle && visible()) draw();
      });
    }
    function visible() { var r = box.getBoundingClientRect(); return box.offsetParent && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth; }
    function stopIdle() { idle = false; hint.classList.add('fade'); }

    // pointer: left-drag rotate, right / shift / two-finger drag pan, wheel / pinch zoom
    var ptrs = {}, last = null, pinch = 0;
    canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    canvas.addEventListener('pointerdown', function (e) {
      canvas.setPointerCapture(e.pointerId); ptrs[e.pointerId] = [e.clientX, e.clientY];
      last = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; pinch = 0; stopIdle();
    });
    canvas.addEventListener('pointermove', function (e) {
      if (!ptrs[e.pointerId]) return;
      ptrs[e.pointerId] = [e.clientX, e.clientY];
      var ids = Object.keys(ptrs);
      if (ids.length === 2) {
        var a = ptrs[ids[0]], b = ptrs[ids[1]], dd = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch) st.dist = Math.max(D.radius * 0.05, st.dist * pinch / dd);
        pinch = dd;
        var mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        if (last && last.two) panBy(mx - last.x, my - last.y);
        last = { x: mx, y: my, two: true }; draw(); return;
      }
      if (!last || last.two) { last = { x: e.clientX, y: e.clientY, pan: e.shiftKey }; return; }
      var dx = e.clientX - last.x, dy = e.clientY - last.y; last.x = e.clientX; last.y = e.clientY;
      if (last.pan) panBy(dx, dy);
      else { st.az -= dx * 0.008; st.el = Math.max(-1.45, Math.min(1.45, st.el + dy * 0.008)); }
      draw();
    });
    function up_(e) { delete ptrs[e.pointerId]; last = null; pinch = 0; }
    canvas.addEventListener('pointerup', up_); canvas.addEventListener('pointercancel', up_);
    function panBy(dx, dy) {
      var F = eyeFrame(), k = st.dist / (FOCAL * W / 2);
      st.target = add(st.target, add(mul(F.x, -dx * k), mul(F.y, -dy * k)));
    }
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault(); stopIdle();
      st.dist = Math.max(D.radius * 0.05, st.dist * Math.exp(e.deltaY * (e.deltaMode ? 0.05 : 0.0015))); draw();
    }, { passive: false });
    if ('ResizeObserver' in window) new ResizeObserver(function () { draw(); }).observe(canvas);
    if ('IntersectionObserver' in window) new IntersectionObserver(function () { draw(); }).observe(box);   // resume the idle sweep
    resize(); reset();
    return { reset: reset, draw: draw };
  }

  document.querySelectorAll('.car-slide[data-recon]').forEach(function (sl) {
    var slug = sl.getAttribute('data-recon'), video = sl.querySelector('video');
    var label = (sl.querySelector('figcaption').textContent || '').split('·')[0].trim();
    var media = document.createElement('div'); media.className = 'r3-media';
    video.parentNode.insertBefore(media, video); media.appendChild(video);
    var box = document.createElement('div'); box.className = 'r3-box'; box.hidden = true; media.appendChild(box);
    var bar = document.createElement('div'); bar.className = 'r3-bar'; media.appendChild(bar);
    var go3d = document.createElement('button'); go3d.type = 'button'; go3d.className = 'r3-btn'; go3d.textContent = 'Explore in 3D';
    var rst = document.createElement('button'); rst.type = 'button'; rst.className = 'r3-btn'; rst.textContent = 'Reset view'; rst.hidden = true;
    var back = document.createElement('button'); back.type = 'button'; back.className = 'r3-btn'; back.textContent = 'Back to video'; back.hidden = true;
    bar.appendChild(rst); bar.appendChild(back); bar.appendChild(go3d);
    var viewer = null;
    function mode(three, quiet) {
      sl.classList.toggle('is-3d', three); box.hidden = !three; video.style.visibility = three ? 'hidden' : '';
      go3d.hidden = three; rst.hidden = back.hidden = !three;
      if (three) { video.pause(); if (viewer) viewer.reset(); }
      else if (!quiet) { var p = video.play(); if (p && p.catch) p.catch(function () {}); }
    }
    go3d.addEventListener('click', function () {
      if (viewer) return mode(true);
      go3d.disabled = true; go3d.textContent = 'Loading…';
      loadScene(slug, function (D) {
        go3d.disabled = false; go3d.textContent = 'Explore in 3D';
        if (!D) { go3d.textContent = 'Could not load'; return; }
        box.hidden = false; viewer = Viewer(box, D, label); mode(true);
      });
    });
    rst.addEventListener('click', function () { viewer && viewer.reset(); });
    back.addEventListener('click', function () { mode(false); });
    sl.r3exit = function () { if (sl.classList.contains('is-3d')) mode(false, true); };   // carousel: leaving a slide restores its clip
  });
})();
