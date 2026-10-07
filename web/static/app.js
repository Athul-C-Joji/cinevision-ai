// CineVision AI - page animations. Plain JavaScript, no libraries.
(function () {
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- iris open: switch the hero on right after the first paint
  requestAnimationFrame(function () {
    requestAnimationFrame(function () { document.body.classList.add("loaded"); });
  });

  // ---- split the headline into words so each one can come into focus
  document.querySelectorAll("[data-split]").forEach(function (el) {
    var words = el.textContent.trim().split(/\s+/);
    el.textContent = "";
    words.forEach(function (w, i) {
      var span = document.createElement("span");
      span.className = "word";
      span.style.setProperty("--i", i);
      span.textContent = w;
      el.appendChild(span);
      el.appendChild(document.createTextNode(" "));
    });
  });

  // ---- numbers count up
  document.querySelectorAll("[data-count]").forEach(function (el) {
    var target = parseInt(el.getAttribute("data-count"), 10) || 0;
    if (reduce) { el.textContent = target; return; }
    el.textContent = "0";
    var start = performance.now() + 900;
    var dur = 900;
    function tick(now) {
      var t = Math.min(Math.max((now - start) / dur, 0), 1);
      el.textContent = Math.round(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });

  // ---- chips get an index, used for the staggered flicker-in
  document.querySelectorAll(".chips").forEach(function (box) {
    box.querySelectorAll(".chip").forEach(function (chip, i) {
      chip.style.setProperty("--i", i);
    });
  });

  // ---- rack-focus reveal: mark elements as "in" when scrolled into view
  var revealObs = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) {
        en.target.classList.add("in");
        revealObs.unobserve(en.target);
      }
    });
  }, { threshold: 0.15 });
  document.querySelectorAll(".rf, .shot").forEach(function (el) { revealObs.observe(el); });

  // ---- clips play only while they are on screen
  if (!reduce) {
    var videoObs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target;
        if (en.isIntersecting) {
          var p = v.play();
          if (p && p.catch) p.catch(function () {});
        } else {
          v.pause();
        }
      });
    }, { threshold: 0.5 });
    document.querySelectorAll("video").forEach(function (v) { videoObs.observe(v); });
  }

  // ---- top bar: scene and shot counter, film-perforation progress line
  var total = parseInt(document.body.getAttribute("data-total"), 10) || 0;
  var hudScene = document.getElementById("hud-scene");
  var hudShot = document.getElementById("hud-shot");
  var bar = document.getElementById("progress-bar");

  var shotObs = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) {
        var s = en.target;
        hudScene.textContent = "Scene " + s.getAttribute("data-scene") + " \u00b7 " + s.getAttribute("data-heading");
        hudShot.textContent = "Shot " + s.getAttribute("data-n") + " / " + total;
      }
    });
  }, { rootMargin: "-45% 0px -45% 0px" });
  document.querySelectorAll(".shot").forEach(function (el) { shotObs.observe(el); });

  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var p = max > 0 ? h.scrollTop / max : 0;
    bar.style.transform = "scaleX(" + p + ")";
    if (h.scrollTop < window.innerHeight * 0.4) {
      hudScene.textContent = "Script Planner";
      hudShot.textContent = total + " shots";
    }
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---- cursor highlight inside glass cards
  document.querySelectorAll(".glass").forEach(function (el) {
    el.addEventListener("mousemove", function (e) {
      var r = el.getBoundingClientRect();
      el.style.setProperty("--mx", (e.clientX - r.left) + "px");
      el.style.setProperty("--my", (e.clientY - r.top) + "px");
    });
  });

  // ---- warm light that glides after the cursor
  var flare = document.querySelector(".flare");
  var tx = 0, ty = 0, fx = 0, fy = 0, flareOn = false;
  function flareLoop() {
    fx += (tx - fx) * 0.08;
    fy += (ty - fy) * 0.08;
    flare.style.transform = "translate(" + fx + "px," + fy + "px)";
    requestAnimationFrame(flareLoop);
  }
  if (flare && !reduce && window.matchMedia("(hover: hover)").matches) {
    window.addEventListener("mousemove", function (e) {
      tx = e.clientX;
      ty = e.clientY;
      if (!flareOn) {
        flareOn = true;
        fx = tx;
        fy = ty;
        flare.style.opacity = 1;
        requestAnimationFrame(flareLoop);
      }
    });
  }
})();