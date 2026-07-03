var BACKEND_URL = "https://cinechat-u65j.onrender.com";

var appMode = "before";
var appMovie = "";
var appHistory = [];
var appLang = "az";
var appTheme = "ocean-dark";
var watchlist = [];
var recognition = null;
var isListening = false;

try {
  var saved = JSON.parse(localStorage.getItem("cc_app") || "{}");
  if (saved.mode)    appMode    = saved.mode;
  if (saved.movie)   appMovie   = saved.movie;
  if (saved.history) appHistory = saved.history;
  if (saved.lang)    appLang    = saved.lang;
  if (saved.theme)   appTheme   = saved.theme;
} catch(e) {}

try { watchlist = JSON.parse(localStorage.getItem("cc_watchlist") || "[]"); } catch(e) {}

function saveApp() {
  localStorage.setItem("cc_app", JSON.stringify({
    mode: appMode, movie: appMovie,
    history: appHistory, lang: appLang, theme: appTheme
  }));
}

// initApp is called from index.html button onclick
function initApp() {
  document.documentElement.setAttribute("data-theme", appTheme);
  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) themeBtn.textContent = appTheme === "ocean-dark" ? "🌙" : "☀️";
  var lbAz    = document.getElementById("lb-az");
  var lbEn    = document.getElementById("lb-en");
  if (lbAz)   lbAz.classList.toggle("active",     appLang === "az");
  if (lbEn)   lbEn.classList.toggle("active",     appLang === "en");
  updateAllUI();
  renderWatchlist();
  if (appHistory.length === 0) {
    addBubble("bot", getWelcome(), false, false, false);
  } else {
    for (var i = 0; i < appHistory.length; i++) {
      var m = appHistory[i];
      if (m.role === "user")      addBubble("user", m.content, false, false, false);
      else if (m.role === "assistant") addBubble("bot", m.content, false, true, true);
    }
    if (appMovie) fetchSuggestions(appMovie);
  }
  renderQuickBtns();
}

function getWelcome() {
  if (appLang === "az") return "Salam! Film adi yazin — izlemeden evvel spoilersiz melumat alin, izledikden sonra derin analiz aparaq.";
  return "Hello! Type a movie name to get spoiler-free info or deep analysis after watching.";
}

function toggleTheme() {
  setTheme(appTheme === "ocean-dark" ? "ocean-light" : "ocean-dark");
}

function setTheme(t) {
  appTheme = t;
  document.documentElement.setAttribute("data-theme", t);
  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) themeBtn.textContent = t === "ocean-dark" ? "🌙" : "☀️";
  saveApp();
}

function setLang(l) {
  appLang = l;
  var lbAz = document.getElementById("lb-az");
  var lbEn = document.getElementById("lb-en");
  if (lbAz) lbAz.classList.toggle("active", l === "az");
  if (lbEn) lbEn.classList.toggle("active", l === "en");
  updateAllUI();
  saveApp();
}

function getText(azText, enText) {
  return appLang === "az" ? azText : enText;
}

function updateAllUI() {
  setEl("lbl-lang",      getText("Dil",               "Language"));
  setEl("lbl-search",    getText("🔍 Film Axtar",     "🔍 Search Movie"));
  setEl("lbl-watchlist", getText("📋 Watchlist",      "📋 Watchlist"));
  setEl("lbl-notify",    getText("📧 Yeni Film Bildirisi", "📧 New Movie Alert"));
  setEl("notify-btn",    getText("Abune ol",          "Subscribe"));
  setEl("lbl-before",    getText("Evvel",             "Before"));
  setEl("lbl-after",     getText("Sonra",             "After"));
  setEl("chat-sub", appMode === "before"
    ? getText("Spoilersiz rejim", "Spoiler-free mode")
    : getText("Analiz rejimi",   "Analysis mode"));
  setAttr("search-input", "placeholder", getText("Film adi...",            "Movie name..."));
  setAttr("notify-email", "placeholder", "email@example.com");
  setAttr("chat-input",   "placeholder", getText("Film adi ve ya sual...", "Movie name or question..."));
  var btnB = document.getElementById("btn-before");
  var btnA = document.getElementById("btn-after");
  if (btnB) btnB.classList.toggle("active", appMode === "before");
  if (btnA) btnA.classList.toggle("active", appMode === "after");
  renderQuickBtns();
  renderWatchlist();
}

function setEl(id, val) {
  var el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setAttr(id, attr, val) {
  var el = document.getElementById(id);
  if (el) el.setAttribute(attr, val);
}

function setMode(m) {
  appMode = m;
  setEl("chat-sub", m === "before"
    ? getText("Spoilersiz rejim", "Spoiler-free mode")
    : getText("Analiz rejimi",   "Analysis mode"));
  var btnB = document.getElementById("btn-before");
  var btnA = document.getElementById("btn-after");
  if (btnB) btnB.classList.toggle("active", m === "before");
  if (btnA) btnA.classList.toggle("active", m === "after");
  renderQuickBtns();
  saveApp();
  if (appMovie) {
    addBubble("bot",
      m === "before"
        ? getText("Spoilersiz rejime kecildi.", "Switched to spoiler-free mode.")
        : getText("Analiz rejimine kecildi!", "Switched to analysis mode!"),
      false, false, false);
  }
}

function getQuickBtns() {
  if (appMode === "before") {
    return appLang === "az"
      ? ["Umumi melumat", "Janri nedir?", "Aktyor heyeti", "Izlemeyе deyermi?", "Watchlist-e elave et", "Sohbeti temizle"]
      : ["General info",  "What genre?",  "Cast & crew",   "Worth watching?",   "Add to Watchlist",    "Clear chat"];
  } else {
    return appLang === "az"
      ? ["Esas mesaj",    "Personaj tehlili", "Simvolik sehneler", "Rejissorun niyyeti", "Watchlist-e elave et", "Sohbeti temizle"]
      : ["Main message",  "Character analysis", "Symbolic scenes", "Director's intent",  "Add to Watchlist",    "Clear chat"];
  }
}

function renderQuickBtns() {
  var el = document.getElementById("quick-btns");
  if (!el) return;
  var btns = getQuickBtns();
  var html = "";
  for (var i = 0; i < btns.length; i++) {
    html += '<button class="quick-btn" onclick="handleQuick(\'' + btns[i].replace(/'/g, "\\'") + '\')">' + btns[i] + '</button>';
  }
  el.innerHTML = html;
}

function handleQuick(text) {
  var wlKey = appLang === "az" ? "Watchlist-e elave et" : "Add to Watchlist";
  var clKey = appLang === "az" ? "Sohbeti temizle"      : "Clear chat";
  if (text === wlKey) { addToWatchlist(); return; }
  if (text === clKey) { doClearChat();    return; }
  var ci = document.getElementById("chat-input");
  if (ci) ci.value = text;
  sendMsg();
}

function addToWatchlist() {
  if (!appMovie || watchlist.indexOf(appMovie) !== -1) return;
  watchlist.push(appMovie);
  localStorage.setItem("cc_watchlist", JSON.stringify(watchlist));
  renderWatchlist();
  addBubble("bot",
    getText('"' + appMovie + '" watchlist-e elave edildi!',
            '"' + appMovie + '" added to watchlist!'),
    false, false, false);
}

function removeFromWatchlist(movie) {
  watchlist = watchlist.filter(function(m){ return m !== movie; });
  localStorage.setItem("cc_watchlist", JSON.stringify(watchlist));
  renderWatchlist();
}

function renderWatchlist() {
  var el = document.getElementById("watchlist-items");
  if (!el) return;
  if (watchlist.length === 0) {
    el.innerHTML = '<div class="sb-empty">—</div>';
    return;
  }
  var html = "";
  for (var i = 0; i < watchlist.length; i++) {
    var mv = watchlist[i];
    html += '<div class="wl-item" onclick="startWithMovie(\'' + mv.replace(/'/g, "\\'") + '\')">';
    html += '<span class="wl-name">' + mv + '</span>';
    html += '<button class="wl-del" onclick="event.stopPropagation();removeFromWatchlist(\'' + mv.replace(/'/g, "\\'") + '\')">✕</button>';
    html += '</div>';
  }
  el.innerHTML = html;
}

function doSearch() {
  var inp   = document.getElementById("search-input");
  var query = inp ? inp.value.trim() : "";
  if (!query) return;
  var el = document.getElementById("search-results");
  if (!el) return;
  el.innerHTML = '<div class="sb-empty">...</div>';

  fetch(BACKEND_URL + "/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: query })
  })
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.found) {
        var poster = (d.poster && d.poster !== "N/A")
          ? '<img src="' + d.poster + '" style="width:30px;height:42px;object-fit:cover;border-radius:4px;flex-shrink:0"/>'
          : '<div style="width:30px;height:42px;background:var(--bg-hover);border-radius:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center">🎬</div>';
        var genre   = d.genre || "";
        var rating  = d.rating && d.rating !== "N/A" ? "⭐ " + d.rating : "";
        el.innerHTML =
          '<div class="search-result" onclick="startWithMovie(\'' + d.title.replace(/'/g, "\\'") + '\')">' +
          poster +
          '<div><div class="result-title">' + d.title + ' (' + d.year + ')</div>' +
          '<div class="result-meta">' + rating + (rating ? ' · ' : '') + genre + '</div></div></div>';
      } else {
        el.innerHTML = '<div class="sb-empty">' + getText("Film tapilmadi.", "Movie not found.") + '</div>';
      }
    })
    .catch(function() {
      el.innerHTML = '<div class="sb-empty" style="color:#e55">Network error</div>';
    });
}

function startWithMovie(title) {
  doClearChat(true);
  var ci = document.getElementById("chat-input");
  var si = document.getElementById("search-input");
  var sr = document.getElementById("search-results");
  if (ci) ci.value = title;
  if (si) si.value = "";
  if (sr) sr.innerHTML = "";
  sendMsg();
}

function subscribeNotify() {
  var email = document.getElementById("notify-email").value.trim();
  if (!email || email.indexOf("@") === -1) {
    alert(getText("Duzgun email daxil edin", "Enter a valid email address"));
    return;
  }
  var btn = document.getElementById("notify-btn");
  btn.textContent = "...";
  setTimeout(function() {
    btn.textContent = getText("Abune olundu! ✓", "Subscribed! ✓");
    setTimeout(function() {
      btn.textContent = getText("Abune ol", "Subscribe");
    }, 2500);
  }, 800);
}

// ── Film oxsar tovsiyyeler ──
function fetchSuggestions(movie) {
  fetch(BACKEND_URL + "/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ movie: movie, lang: appLang })
  })
  .then(function(r){ return r.json(); })
  .then(function(data) {
    var films = (data.films || []).slice(0, 4);
    if (films.length > 0) {
      var title = appLang === "az" ? "Oxsar filmler:" : "Similar movies:";
      var html  = '<div class="sug-title">' + title + '</div><div class="sug-chips">';
      for (var i = 0; i < films.length; i++) {
        html += '<button class="sug-chip" onclick="startWithMovie(\'' + films[i].replace(/'/g, "\\'") + '\')">' + films[i] + '</button>';
      }
      html += '</div>';
      var sg = document.getElementById("suggestions");
      if (sg) sg.innerHTML = html;
    }
  })
  .catch(function(){});
}

// ── OMDB poster karti ──
function renderMovieCard(d) {
  if (!d.found) return;
  var box = document.getElementById("messages");
  if (!box) return;
  var wrap = document.createElement("div"); wrap.className = "msg bot";
  var av   = document.createElement("div"); av.className = "avatar bot"; av.textContent = "C";
  var card = document.createElement("div"); card.className = "movie-card";

  var posterHTML = (d.poster && d.poster !== "N/A")
    ? '<img src="' + d.poster + '" alt="' + d.title + '" loading="lazy" style="width:100%"/>'
    : '<div style="width:100%;height:120px;background:var(--bg-hover);display:flex;align-items:center;justify-content:center;font-size:36px">🎬</div>';

  card.innerHTML = posterHTML +
    '<div class="movie-card-body">' +
    '<div class="mc-title">' + d.title + ' (' + d.year + ')</div>' +
    '<div class="mc-meta">' +
      (d.rating && d.rating !== "N/A" ? '⭐ ' + d.rating + ' · ' : '') +
      (d.genre || '') +
      (d.runtime && d.runtime !== "N/A" ? ' · ' + d.runtime : '') +
    '</div>' +
    (d.director && d.director !== "N/A" ? '<div class="mc-meta">🎬 ' + d.director + '</div>' : '') +
    '</div>';

  wrap.appendChild(av);
  wrap.appendChild(card);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

function fetchPoster(title) {
  fetch(BACKEND_URL + "/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title })
  })
    .then(function(r){ return r.json(); })
    .then(renderMovieCard)
    .catch(function(){});
}

// ── Ses tanima ──
function toggleVoice() {
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    alert(getText("Ses ucun Chrome brauzerinden istifade edin.", "Use Chrome browser for voice input."));
    return;
  }
  var mic   = document.getElementById("mic-btn");
  var input = document.getElementById("chat-input");
  if (isListening) {
    if (recognition) recognition.stop();
    return;
  }
  recognition = new SR();
  recognition.lang = appLang === "az" ? "az-AZ" : "en-US";
  recognition.interimResults = true;
  recognition.continuous = false;
  isListening = true;
  mic.classList.add("active");
  if (input) input.placeholder = appLang === "az" ? "Danisın..." : "Listening...";

  recognition.onresult = function(e) {
    var interim = "", final = "";
    for (var i = e.resultIndex; i < e.results.length; i++) {
      var t = e.results[i][0].transcript;
      if (e.results[i].isFinal) final += t; else interim += t;
    }
    if (input) input.value = final || interim;
  };

  recognition.onend = function() {
    mic.classList.remove("active");
    if (input) input.placeholder = getText("Film adi ve ya sual...", "Movie name or question...");
    isListening = false;
    if (input && input.value.trim()) setTimeout(function(){ sendMsg(); }, 400);
  };

  recognition.onerror = function(e) {
    mic.classList.remove("active");
    if (input) input.placeholder = getText("Film adi ve ya sual...", "Movie name or question...");
    isListening = false;
    if (e.error !== "aborted") {
      addBubble("bot", getText("Ses taninmadi.", "Not recognized."), false, false, false);
    }
  };

  recognition.start();
}

// ── Mesaj gonder ──
function sendMsg() {
  var input = document.getElementById("chat-input");
  if (!input) return;
  var text = input.value.trim();
  if (!text) return;
  input.value = "";

  var isFirst = !appMovie;
  if (isFirst) appMovie = text; // ilk mesaj hemise film adi kimi qebul olunur

  addBubble("user", text, false, false, false);
  showTyping();

  // FIX 2: history-ni cavabdan SONRA push et (sira duzgun olsun)
  callAI(text)
    .then(function(reply) {
      removeTyping();
      var spoiler = appMode === "after" && appHistory.length > 0;
      addBubble("bot", reply, spoiler, true, true);
      appHistory.push({ role: "user",      content: text  });
      appHistory.push({ role: "assistant", content: reply });
      saveApp();

      // FIX 5: her mesajda yoxla — yazilan yeni bir film adidirsa
      // poster ve tovsiyeleri yenile (yalniz ilk mesajda deyil)
      checkMovieSwitch(text, isFirst);
    })
    .catch(function(err) {
      removeTyping();
      addBubble("bot",
        getText("Xeta bas verdi: ", "Error: ") + (err.message || "Unknown error"),
        false, false, false);
    });
}

// ── Yazilan metnin (yeni) bir film adi olub-olmadigini yoxlayir ──
function checkMovieSwitch(text, isFirst) {
  fetch(BACKEND_URL + "/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: text })
  })
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.found) {
        var isNewMovie = isFirst || d.title.toLowerCase() !== appMovie.toLowerCase();
        if (isNewMovie) {
          appMovie = d.title;
          renderMovieCard(d);
          fetchSuggestions(d.title);
          saveApp();
        }
      } else if (isFirst) {
        // Ilk mesaj OMDB-de tapilmadi, amma yene de tovsiyye cehdi edek
        fetchSuggestions(appMovie);
      }
    })
    .catch(function(){});
}

// ── Backend chat sorgusu ──
function callAI(userMsg) {
  return fetch(BACKEND_URL + "/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: userMsg,
      history: appHistory.slice(-12),
      mode: appMode,
      lang: appLang,
      movie: appMovie
    })
  })
  .then(function(r) {
    if (!r.ok) {
      return r.json().then(function(e) {
        var msg = (e.detail) ? e.detail : ("HTTP " + r.status);
        throw new Error(msg);
      }).catch(function(err) {
        if (err.message) throw err;
        throw new Error("HTTP " + r.status);
      });
    }
    return r.json();
  })
  .then(function(data) {
    if (typeof data.reply !== "string") {
      throw new Error(getText("Bos cavab alindi.", "Empty response received."));
    }
    return data.reply;
  });
}

// ── Sohheti temizle ──
function doClearChat(silent) {
  appHistory = [];
  appMovie   = "";
  saveApp();
  var msgs = document.getElementById("messages");
  var sugs = document.getElementById("suggestions");
  if (msgs) msgs.innerHTML = "";
  if (sugs) sugs.innerHTML = "";
  if (!silent) addBubble("bot", getWelcome(), false, false, false);
}

// ── Mesaj balonu ──
function addBubble(role, text, spoiler, showActions, isHTML) {
  var box = document.getElementById("messages");
  if (!box) return;

  var wrap = document.createElement("div"); wrap.className = "msg " + role;
  var av   = document.createElement("div"); av.className = "avatar " + role;
  av.textContent = role === "bot" ? "C" : "S";

  var bbl = document.createElement("div"); bbl.className = "bubble " + role;

  if (spoiler) {
    var badge = document.createElement("div"); badge.className = "spoiler-badge";
    badge.textContent = appLang === "az" ? "Spoiler movcuddur" : "Contains spoilers";
    bbl.appendChild(badge);
    bbl.appendChild(document.createElement("br"));
  }

  if (isHTML) {
    var d = document.createElement("div"); d.innerHTML = text; bbl.appendChild(d);
  } else {
    bbl.appendChild(document.createTextNode(text));
  }

  if (role === "bot" && showActions) {
    var acts = document.createElement("div"); acts.className = "bubble-actions";
    var like = document.createElement("button"); like.className = "act-btn"; like.textContent = "👍";
    var dis  = document.createElement("button"); dis.className  = "act-btn"; dis.textContent  = "👎";
    like.onclick = function(){ like.classList.toggle("liked"); dis.classList.remove("disliked"); };
    dis.onclick  = function(){ dis.classList.toggle("disliked"); like.classList.remove("liked"); };

    var stars = document.createElement("div"); stars.className = "stars";
    for (var i = 1; i <= 5; i++) {
      (function(val) {
        var s = document.createElement("button");
        s.className   = "star";
        // FIX 5: * yerine duzgun ulduz emoji
        s.textContent = "★";
        s.onclick = function() {
          var all = stars.querySelectorAll(".star");
          for (var j = 0; j < all.length; j++) all[j].classList.toggle("lit", j < val);
        };
        stars.appendChild(s);
      })(i);
    }

    acts.appendChild(like);
    acts.appendChild(dis);
    acts.appendChild(stars);
    bbl.appendChild(acts);
  }

  wrap.appendChild(av);
  wrap.appendChild(bbl);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

// ── Typing animasiyasi ──
function showTyping() {
  var box = document.getElementById("messages"); if (!box) return;
  var w   = document.createElement("div"); w.className = "msg bot"; w.id = "typing-ind";
  var av  = document.createElement("div"); av.className = "avatar bot"; av.textContent = "C";
  var bbl = document.createElement("div"); bbl.className = "bubble bot typing";
  bbl.innerHTML = "<div class='dot'></div><div class='dot'></div><div class='dot'></div>";
  w.appendChild(av);
  w.appendChild(bbl);
  box.appendChild(w);
  box.scrollTop = box.scrollHeight;
}

function removeTyping() {
  var el = document.getElementById("typing-ind"); if (el) el.remove();
}

// ── Tema saxla ──
document.documentElement.setAttribute("data-theme", appTheme);
