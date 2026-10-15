// Renders LaTeX `algorithm`/`algorithmic` blocks (<pre class="pseudocode">) with
// pseudocode.js. Documenter loads KaTeX through RequireJS as the module "katex";
// pseudocode.js is loaded through the same RequireJS so it reuses that KaTeX
// (no second copy, nothing global disturbed). The rendered block inherits the
// page's text colour, so it follows the light and dark themes.
(function () {
  var PSEUDO = "https://cdn.jsdelivr.net/npm/pseudocode@2.4.1/build/pseudocode.min";
  var PSEUDO_CSS = "https://cdn.jsdelivr.net/npm/pseudocode@2.4.1/build/pseudocode.min.css";

  // require.js is fetched from a CDN, and Documenter registers the "katex" path
  // only once its own documenter.js has run. Requesting "katex" before that
  // makes RequireJS fetch assets/katex.js (404) and mark the module failed,
  // which also breaks Documenter's math rendering. So wait (up to 20 s) until
  // the path is registered before asking for anything.
  var tries = 0;
  function katexRegistered() {
    try {
      var cfg = requirejs.s.contexts._.config;
      return !!(cfg && cfg.paths && cfg.paths.katex);
    } catch (e) { return false; }
  }
  function start() {
    if (!document.querySelector("pre.pseudocode")) return;
    if (typeof requirejs === "undefined" || !katexRegistered()) {
      if (tries++ < 200) setTimeout(start, 100);
      return;
    }
    var l = document.createElement("link");
    l.rel = "stylesheet"; l.href = PSEUDO_CSS; document.head.appendChild(l);
    requirejs.config({ paths: { pseudocode: PSEUDO } });
    requirejs(["katex"], function (katex) {
      // pseudocode.js typesets with the global `katex`.
      window.katex = window.katex || katex;
      requirejs(["pseudocode"], function (pseudocode) {
        var blocks = document.querySelectorAll("pre.pseudocode");
        for (var i = 0; i < blocks.length; i++) {
          try {
            pseudocode.renderElement(blocks[i], {
              lineNumber: true, captionCount: i, noEnd: false,
              commentDelimiter: "▷"
            });
          } catch (e) { console.error("pseudocode.js:", e); }
        }
      });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
