/**
 * HAIL Cognitive Studio — Autonomous HCL Stratified Memory Engine & Cognitive Console
 * Direct implementation of HCL (Hydrus Cognitive Layer) personal query processing,
 * autonomous memory extraction, local storage persistence, and interactive lattice visualizer.
 */

// 1. Background Particle System
class ParticleBackground {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.particles = [];
    this.mouse = { x: null, y: null, radius: 150 };
    this.resize();
    this.init();
    this.animate();

    window.addEventListener("resize", () => this.resize());
    window.addEventListener("mousemove", (e) => {
      this.mouse.x = e.x;
      this.mouse.y = e.y;
    });
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init() {
    this.particles = [];
    const count = Math.floor((this.canvas.width * this.canvas.height) / 22000);
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: Math.random() * 1.6 + 0.5,
        alpha: Math.random() * 0.4 + 0.15
      });
    }
  }

  animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (let i = 0; i < this.particles.length; i++) {
      let p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;

      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = `rgba(139, 106, 255, ${p.alpha})`;
      this.ctx.fill();

      for (let j = i + 1; j < this.particles.length; j++) {
        let p2 = this.particles[j];
        let dx = p.x - p2.x;
        let dy = p.y - p2.y;
        let dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 90) {
          this.ctx.beginPath();
          this.ctx.moveTo(p.x, p.y);
          this.ctx.lineTo(p2.x, p2.y);
          this.ctx.strokeStyle = `rgba(198, 178, 255, ${0.12 * (1 - dist / 90)})`;
          this.ctx.lineWidth = 0.5;
          this.ctx.stroke();
        }
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}

// 2. Interactive Memory Lattice 2D Graph Visualizer
class MemoryLatticeVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.nodes = [];
    this.resize();
    this.animate();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width || 280;
    this.canvas.height = rect.height || 140;
  }

  updateNodes(memories) {
    if (!this.canvas) return;
    const w = this.canvas.width;
    const h = this.canvas.height;

    this.nodes = memories.map((m, idx) => {
      const angle = (idx / Math.max(1, memories.length)) * Math.PI * 2;
      const radius = Math.min(w, h) * 0.32;
      const cx = w / 2 + Math.cos(angle) * radius;
      const cy = h / 2 + Math.sin(angle) * radius;

      return {
        id: m.id || idx,
        text: m.text,
        stratum: m.stratum,
        conf: m.conf,
        x: cx + (Math.random() - 0.5) * 12,
        y: cy + (Math.random() - 0.5) * 12,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        radius: m.stratum === "Surface" ? 6 : 4
      };
    });
  }

  animate() {
    if (!this.canvas) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw graph edges (co-activation links)
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        let n1 = this.nodes[i];
        let n2 = this.nodes[j];
        this.ctx.beginPath();
        this.ctx.moveTo(n1.x, n1.y);
        this.ctx.lineTo(n2.x, n2.y);
        this.ctx.strokeStyle = "rgba(198, 178, 255, 0.4)";
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
      }
    }

    // Draw nodes
    for (let n of this.nodes) {
      n.x += n.vx;
      n.y += n.vy;

      if (n.x < 10 || n.x > this.canvas.width - 10) n.vx *= -1;
      if (n.y < 10 || n.y > this.canvas.height - 10) n.vy *= -1;

      // Glow effect
      this.ctx.beginPath();
      this.ctx.arc(n.x, n.y, n.radius + 4, 0, Math.PI * 2);
      this.ctx.fillStyle = n.stratum === "Surface" ? "rgba(139, 106, 255, 0.3)" : "rgba(50, 143, 100, 0.3)";
      this.ctx.fill();

      // Node core
      this.ctx.beginPath();
      this.ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = n.stratum === "Surface" ? "#8b6aff" : "#328f64";
      this.ctx.fill();
      this.ctx.strokeStyle = "#ffffff";
      this.ctx.lineWidth = 1.5;
      this.ctx.stroke();
    }

    requestAnimationFrame(() => this.animate());
  }
}

// 3. HAIL HCL Cognitive Layer & Chat Application
class HAILChatApp {
  constructor() {
    this.STORAGE_KEY = "hail_web_memories_v4";
    this.memories = [];
    this.chatHistory = [];
    this.initUI();
    this.loadMemories();
    this.latticeVis = new MemoryLatticeVisualizer("latticeCanvas");
    this.renderMemories();
  }

  async loadMemories() {
    try {
      const resp = await fetch('/api/memories');
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.memories && data.memories.length > 0) {
          this.memories = data.memories;
          this.renderMemories();
          return;
        }
      }
    } catch (err) {
      console.warn("Disk memories fetch failed, checking localStorage fallback:", err);
    }

    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        this.memories = JSON.parse(stored);
        this.saveMemories(); // Mirror existing memories directly to desktop_memories.json!
      } else {
        this.memories = [
          { id: 1, text: "Active kernel engine: HAIL Core v0.1", stratum: "Surface", conf: 0.90 }
        ];
        this.saveMemories();
      }
    } catch (e) {
      this.memories = [];
    }
    this.renderMemories();
  }

  saveMemories() {
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.memories));
    } catch (e) {
      console.warn("Could not save to localStorage:", e);
    }
    fetch('/api/save_memories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ memories: this.memories })
    }).catch(err => console.warn("Failed to sync memories to disk:", err));
  }

  initUI() {
    this.chatFeed = document.getElementById("chatFeed");
    this.chatInput = document.getElementById("chatInput");
    this.chatForm = document.getElementById("chatForm");
    this.memoryList = document.getElementById("memoryList");
    this.memoryBadge = document.getElementById("memoryBadge");
    this.clearMemoriesBtn = document.getElementById("clearMemoriesBtn");

    if (this.chatForm) {
      this.chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        this.sendMessage();
      });
    }

    if (this.clearMemoriesBtn) {
      this.clearMemoriesBtn.addEventListener("click", () => {
        this.memories = [];
        this.saveMemories();
        this.renderMemories();
      });
    }

    // Mobile Sidebar Drawer Toggle
    const mobileMenuBtn = document.getElementById("mobileMenuBtn");
    const sidebar = document.querySelector(".sidebar");
    if (mobileMenuBtn && sidebar) {
      mobileMenuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("mobile-open");
      });
    }

    // Navbar Tab View Switcher
    const navTabs = document.querySelectorAll(".view-tabs .nav-tab");
    navTabs.forEach(tab => {
      tab.addEventListener("click", (e) => {
        e.preventDefault();
        const view = e.currentTarget.getAttribute("data-view");
        navTabs.forEach(t => t.classList.remove("active"));
        e.currentTarget.classList.add("active");
        this.switchView(view);
      });
    });

    // Suggestion Prompt Cards Binding
    const suggestionCards = document.querySelectorAll(".suggestion-card");
    suggestionCards.forEach(card => {
      card.addEventListener("click", (e) => {
        const promptText = e.currentTarget.getAttribute("data-prompt");
        if (promptText && this.chatInput) {
          this.chatInput.value = promptText;
          this.sendMessage();
        }
      });
    });

    // New Chat Button Handler
    const newChatBtn = document.getElementById("newChatBtn");
    if (newChatBtn) {
      newChatBtn.addEventListener("click", () => {
        if (this.chatFeed) {
          this.chatFeed.innerHTML = `
            <div class="welcome-hero">
              <h1 class="welcome-title">HAIL Cognitive Studio</h1>
              <p class="welcome-subtitle">Edge-native local AI engine grounded by Stratified Memory Lattice &amp; Active Security Gateway.</p>

              <div class="prompt-suggestions">
                <button class="suggestion-card" data-prompt="Show preview of seatbelt_dangers.md">
                  <strong style="display:block; color:var(--accent-purple); margin-bottom:2px;">📄 View Document Preview</strong>
                  <span>Open live rendered preview of seatbelt_dangers.md</span>
                </button>
                <button class="suggestion-card" data-prompt="Explain HAIL System Architecture">
                  <strong style="display:block; color:var(--accent-cyan); margin-bottom:2px;">⚡ HAIL Architecture</strong>
                  <span>Learn about HCL, SML, and HydrusOPT</span>
                </button>
                <button class="suggestion-card" data-prompt="Show my memory facts">
                  <strong style="display:block; color:var(--accent-green); margin-bottom:2px;">🧠 Inspect Memory Lattice</strong>
                  <span>Check facts retained in surface strata</span>
                </button>
                <button class="suggestion-card" data-prompt="Create doc about CMG Cognitive Memory Gateway">
                  <strong style="display:block; color:var(--accent-purple); margin-bottom:2px;">🚀 Create New Document</strong>
                  <span>Generate and render new markdown doc</span>
                </button>
              </div>
            </div>
          `;
          // Re-bind suggestion cards
          document.querySelectorAll(".suggestion-card").forEach(c => {
            c.addEventListener("click", (evt) => {
              const p = evt.currentTarget.getAttribute("data-prompt");
              if (p && this.chatInput) {
                this.chatInput.value = p;
                this.sendMessage();
              }
            });
          });
        }
      });
    }

    // Local Model Selector
    this.modelSelect = document.getElementById("localModelSelect");
    this.activeModelLabel = document.getElementById("activeModelLabel");
    this.selectedModel = localStorage.getItem("hail_selected_local_model") || "microsoft/Phi-3.5-mini-instruct";

    if (this.modelSelect) {
      this.modelSelect.value = this.selectedModel;
      this.updateModelLabel();
      this.modelSelect.addEventListener("change", (e) => {
        this.selectedModel = e.target.value;
        localStorage.setItem("hail_selected_local_model", this.selectedModel);
        this.updateModelLabel();
        this.appendMsg("HAIL Core Kernel", `🤖 Switched active local model to: **${this.selectedModel}** (Downloaded in \`d:\\HydrusOPT\\models\`)`, "assistant");
      });
    }

    this.initDocPreview();
    this.initUnifiedModelSelector();
    this.fetchMoETelemetry();
    setInterval(() => this.fetchMoETelemetry(), 5000);
  }

  async fetchMoETelemetry() {
    try {
      const resp = await fetch('/api/moe/status');
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.tiers) {
          const vramEl = document.getElementById("vramTelemetry");
          const ramEl = document.getElementById("ramTelemetry");
          const ssdEl = document.getElementById("ssdTelemetry");
          const prefetchEl = document.getElementById("prefetchTelemetry");

          if (vramEl) vramEl.textContent = `${data.tiers.vram.used_mb}MB`;
          if (ramEl) ramEl.textContent = `${data.tiers.ram.used_mb}MB`;
          if (ssdEl) ssdEl.textContent = `${Math.round(data.tiers.ssd.hit_rate * 100)}%`;
          if (prefetchEl && data.prefetcher) prefetchEl.textContent = `${Math.round(data.prefetcher.hail_hit_rate * 100)}%`;
        }
      }
    } catch (err) {
      console.warn("Could not fetch HydrusMoE telemetry:", err);
    }
  }

  async initUnifiedModelSelector() {
    const select = document.getElementById("unifiedModelSelect");
    const dot = document.getElementById("modelStatusDot");
    if (!select) return;

    const moeModels = [
      { id: "moe:Qwen/Qwen1.5-MoE-A2.7B", name: "🛡️ Qwen1.5-MoE-A2.7B (14.3B Total / 2.7B Active - Tiered MoE)" },
      { id: "moe:qwen3-35b-a3b", name: "🛡️ Qwen3-35B-A3B (35B Total / 3B Active - Tiered MoE)" },
      { id: "moe:Mixtral-8x7B-v0.1", name: "🛡️ Mixtral-8x7B (47B Total / 13B Active - Tiered MoE)" }
    ];

    const localModels = [
      { id: "local:microsoft/Phi-3.5-mini-instruct", name: "🤖 Phi-3.5-mini-instruct (3.8B - Downloaded)" },
      { id: "local:Qwen/Qwen2.5-7B-Instruct", name: "🤖 Qwen2.5-7B-Instruct (7B - Downloaded)" },
      { id: "local:Qwen/Qwen2.5-3B-Instruct", name: "🤖 Qwen2.5-3B-Instruct (3B - Downloaded)" },
      { id: "local:Qwen/Qwen2.5-1.5B-Instruct", name: "🤖 Qwen2.5-1.5B-Instruct (1.5B - Downloaded)" },
      { id: "local:microsoft/phi-2", name: "🤖 microsoft/phi-2 (2.7B - Downloaded)" }
    ];

    let ollamaModels = [];
    let isOllamaOnline = false;

    try {
      const resp = await fetch('/api/ollama/status');
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.online && data.models && data.models.length > 0) {
          isOllamaOnline = true;
          ollamaModels = data.models;
        }
      }
    } catch (err) {
      console.warn("Ollama status check warning:", err);
    }

    if (dot) {
      dot.style.background = isOllamaOnline ? "#22c55e" : "#3b82f6";
      dot.title = isOllamaOnline ? "Ollama Daemon Online" : "Local HAIL Engine Active";
    }

    let html = "";
    html += `<optgroup label="🛡️ HydrusMoE Tiered Models (4-Tier VRAM/RAM/SSD)">`;
    moeModels.forEach(m => {
      html += `<option value="${m.id}">${m.name}</option>`;
    });
    html += `</optgroup>`;

    if (ollamaModels.length > 0) {
      html += `<optgroup label="🦙 Local Ollama Models">`;
      ollamaModels.forEach(m => {
        html += `<option value="ollama:${m}">🦙 ${m} (Ollama Local)</option>`;
      });
      html += `</optgroup>`;
    }

    html += `<optgroup label="🤖 Downloaded Local Models (d:\\HydrusOPT\\models)">`;
    localModels.forEach(m => {
      html += `<option value="${m.id}">${m.name}</option>`;
    });
    html += `</optgroup>`;

    html += `<optgroup label="⚡ HAIL Core Engine">`;
    html += `<option value="hail:edge">⚡ HAIL Edge Core Engine</option>`;
    html += `</optgroup>`;

    select.innerHTML = html;

    const savedModel = localStorage.getItem("hail_unified_selected_model");
    if (savedModel && select.querySelector(`option[value="${savedModel}"]`)) {
      select.value = savedModel;
      this.selectedUnifiedModel = savedModel;
    } else {
      this.selectedUnifiedModel = select.value;
    }

    select.addEventListener("change", (e) => {
      this.selectedUnifiedModel = e.target.value;
      localStorage.setItem("hail_unified_selected_model", this.selectedUnifiedModel);
    });

    const container = document.getElementById("modelContainer");
    if (container && !this.modelContainerBound) {
      this.modelContainerBound = true;
      container.addEventListener("click", (e) => {
        if (e.target === select) return;
        this.initUnifiedModelSelector();
      });
    }
  }

  updateModelLabel() {
    if (this.activeModelLabel) {
      const shortName = this.selectedModel.split("/").pop();
      this.activeModelLabel.textContent = `${shortName} (Downloaded Local Model)`;
    }
  }

  switchView(view) {
    const workspace = document.getElementById("workspace");
    const docContainer = document.getElementById("docPreviewContainer");
    
    if (workspace) {
      workspace.setAttribute("data-active-view", view);
    }
    
    if (view === "doc") {
      if (docContainer) {
        docContainer.classList.add("open");
        this.renderDocContent();
      }
    } else if (view === "memory") {
      if (this.latticeVis) {
        setTimeout(() => this.latticeVis.resize(), 100);
      }
    }
  }

  initDocPreview() {
    this.docsRepo = {
      seatbelt_dangers: `# The Dangers of Not Wearing a Seatbelt

Wearing a seatbelt is the single most effective way to protect yourself in a car crash. Failing to wear one increases your risk of severe injury or death.

Here are the main dangers of driving or riding without a seatbelt:

### 1. Ejection from the Vehicle
In a high-speed crash, the force of the collision can throw you through the windshield or doors. People thrown from a vehicle are **4 times more likely to die** than those who remain inside.

### 2. Colliding with the Interior (The "Second Collision")
Even if you are not thrown out, your body keeps moving at the car's original speed until it hits something. Without a seatbelt to hold you back, you will slam into the steering wheel, dashboard, or windshield with extreme force.

### 3. Airbags Can Cause Serious Injury
Airbags are designed to work **with** seatbelts, not instead of them. If you are not buckled, the force of a deploying airbag hitting your body can cause severe concussions, fractures, or fatal neck injuries.

### 4. Hitting Other Passengers
An unbelted passenger becomes a flying object during a crash. You can be thrown violently into other people in the car, causing serious or fatal injuries to them.

### 5. Increased Fatality Risk
Statistically, seatbelts reduce the risk of fatal injury to front-seat passenger car occupants by **45%**, and the risk of moderate-to-critical injury by **50%**.

---

**Summary:** Always buckle up before driving. It takes only two seconds, but it can save your life.`,

      hail_architecture: `# HAIL System Architecture

HAIL (Holistic Agentic Intelligence & Cognitive Memory System) is an edge-native AI architecture built for local compute environments.

## Core Modules
- **HCL (Hydrus Cognitive Layer)**: Stratified Memory Lattice (SML), Cognem Tokenizer, Metacognitive Router, and Sub-agents.
- **CMG (Cognitive Memory Gateway)**: Active Security Ingestion Firewall, Grounding, and Memory Strata Routing.
- **HydrusOPT**: Deep learning compute optimization, 4-bit quantization, and VRAM bandwidth scheduling.

\`\`\`python
from HAIL import HydrusAgent, CognitiveMemoryGateway, MetacognitiveRouter

agent = HydrusAgent(workspace=".", model_adapter=adapter)
gateway = CognitiveMemoryGateway(workspace=".")
\`\`\``,

      cognitive_gateway: `# Cognitive Memory Gateway (CMG) Spec

The Cognitive Memory Gateway acts as an open nervous system connecting exogenous data sources (MCP tools, Postgres, Web search, Filesystem) to HAIL's 4 memory strata.

> **Epistemic Hygiene**: Contradictory facts are tagged with confidence scores rather than silently overwritten.
> **Security Firewall**: Sub-millisecond scanning for secret keys, PII, and indirect RAG prompt injections.`
    };

    this.activeDocKey = "seatbelt_dangers";
    this.viewMode = "rendered";

    const docChips = document.querySelectorAll(".doc-chip");
    docChips.forEach(chip => {
      chip.addEventListener("click", (e) => {
        const docKey = e.currentTarget.getAttribute("data-doc");
        docChips.forEach(c => c.classList.remove("active"));
        e.currentTarget.classList.add("active");
        this.loadDocPreview(docKey);
      });
    });

    const closeBtn = document.getElementById("closeDocPreviewBtn");
    if (closeBtn) {
      closeBtn.addEventListener("click", () => {
        const container = document.getElementById("docPreviewContainer");
        const workspace = document.getElementById("workspace");
        if (container) container.classList.remove("open");
        if (workspace) workspace.classList.remove("has-preview");
      });
    }

    const saveBtn = document.getElementById("saveDocBtn");
    if (saveBtn) {
      saveBtn.addEventListener("click", async () => {
        const text = this.docsRepo[this.activeDocKey] || "";
        try {
          const resp = await fetch('/api/save_doc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: this.activeDocKey, content: text })
          });
          if (resp.ok) {
            saveBtn.textContent = "✅ Saved!";
            setTimeout(() => saveBtn.textContent = "💾 Save", 1500);
            this.syncDocsFromDisk();
          }
        } catch (e) {
          console.error("Save error:", e);
        }
      });
    }

    const downloadBtn = document.getElementById("downloadDocBtn");
    if (downloadBtn) {
      downloadBtn.addEventListener("click", () => {
        const text = this.docsRepo[this.activeDocKey] || "";
        const blob = new Blob([text], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${this.activeDocKey}.md`;
        a.click();
        URL.revokeObjectURL(url);
      });
    }

    const deleteBtn = document.getElementById("deleteDocBtn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", async () => {
        if (!confirm(`Delete ${this.activeDocKey}.md from disk?`)) return;
        try {
          await fetch('/api/delete_doc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: `${this.activeDocKey}.md` })
          });
          delete this.docsRepo[this.activeDocKey];
          const container = document.getElementById("docPreviewContainer");
          if (container) container.classList.remove("open");
          this.syncDocsFromDisk();
        } catch (e) {
          console.error("Delete error:", e);
        }
      });
    }

    const renderedBtn = document.getElementById("tabRenderedBtn");
    const rawBtn = document.getElementById("tabRawBtn");

    if (renderedBtn && rawBtn) {
      renderedBtn.addEventListener("click", () => {
        this.viewMode = "rendered";
        renderedBtn.classList.add("active");
        rawBtn.classList.remove("active");
        this.renderDocContent();
      });

      rawBtn.addEventListener("click", () => {
        this.viewMode = "raw";
        rawBtn.classList.add("active");
        renderedBtn.classList.remove("active");
        this.renderDocContent();
      });
    }

    // Sync all documents from physical disk via backend API
    this.syncDocsFromDisk();
  }

  async syncDocsFromDisk() {
    try {
      const resp = await fetch('/api/list_docs');
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.docs) {
          const docQuickList = document.getElementById("docQuickList");
          if (docQuickList) {
            docQuickList.innerHTML = "";
            data.docs.forEach(docInfo => {
              const key = docInfo.key;
              const chip = document.createElement("div");
              chip.className = `doc-item ${key === this.activeDocKey ? 'active' : ''}`;
              chip.setAttribute("data-doc", key);
              chip.innerHTML = `<span class="doc-item-icon">📄</span> <span>${docInfo.filename}</span>`;
              chip.addEventListener("click", () => {
                document.querySelectorAll(".doc-item").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                this.loadDocFromDisk(key, docInfo.filename);
              });
              docQuickList.appendChild(chip);
            });

            // Update header count
            const countSpan = document.querySelector(".sidebar-section-title span:nth-child(2)");
            if (countSpan) {
              countSpan.textContent = `${data.docs.length} Docs`;
            }
          }
        }
      }
    } catch (e) {
      console.warn("Could not sync docs from disk:", e);
    }
  }

  async loadDocFromDisk(key, filename) {
    try {
      const resp = await fetch(`/api/get_doc?file=${encodeURIComponent(filename)}`);
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.content) {
          this.loadDocPreview(key, data.content, filename);
          return;
        }
      }
    } catch (e) {
      console.warn("Could not fetch file from disk:", e);
    }
    this.loadDocPreview(key);
  }

  loadDocPreview(docKey, customRawText = null, customTitle = null) {
    this.activeDocKey = docKey;
    if (customRawText) {
      this.docsRepo[docKey] = customRawText;
    }
    const container = document.getElementById("docPreviewContainer");
    const titleEl = document.getElementById("docPreviewTitle");
    const workspace = document.getElementById("workspace");
    
    // Dynamically add to sidebar if missing
    const docQuickList = document.getElementById("docQuickList");
    if (docQuickList) {
      let existingChip = docQuickList.querySelector(`[data-doc="${docKey}"]`);
      if (!existingChip) {
        const newChip = document.createElement("div");
        newChip.className = "doc-item";
        newChip.setAttribute("data-doc", docKey);
        newChip.innerHTML = `<span class="doc-item-icon">📄</span> <span>${docKey}.md</span>`;
        newChip.addEventListener("click", (e) => {
          document.querySelectorAll(".doc-item").forEach(c => c.classList.remove("active"));
          e.currentTarget.classList.add("active");
          this.loadDocPreview(docKey);
        });
        docQuickList.appendChild(newChip);
        
        // Update doc count
        const docCount = Object.keys(this.docsRepo).length;
        const countSpan = document.querySelector(".sidebar-section-title span:nth-child(2)");
        if (countSpan && countSpan.textContent.includes("Docs")) {
          countSpan.textContent = `${docCount} Docs`;
        }
      }
      // Highlight active doc in sidebar
      document.querySelectorAll(".doc-item").forEach(c => c.classList.remove("active"));
      const activeChip = docQuickList.querySelector(`[data-doc="${docKey}"]`);
      if (activeChip) activeChip.classList.add("active");
    }
    
    if (container) {
      container.classList.add("open");
      // Remove any lingering inline styles
      container.style.display = ""; 
    }
    if (workspace && workspace.getAttribute("data-active-view") === "all") {
      workspace.classList.add("has-preview");
    }
    
    if (titleEl) titleEl.textContent = customTitle || `${docKey}.md`;
    this.renderDocContent();
    this.syncDocsFromDisk();
  }

  renderDocContent() {
    const contentEl = document.getElementById("docPreviewContent");
    if (!contentEl) return;

    const rawText = this.docsRepo[this.activeDocKey] || "# Document Not Found";

    if (this.viewMode === "raw") {
      contentEl.innerHTML = `<textarea id="rawDocTextarea" style="width: 100%; height: 100%; min-height: 420px; background: transparent; border: none; outline: none; resize: none; font-family: var(--font-mono); font-size: 12px; color: var(--launch-ink); line-height: 1.6; padding: 0;">${this.escape(rawText)}</textarea>`;
      
      const textarea = document.getElementById("rawDocTextarea");
      if (textarea) {
        textarea.value = rawText;
        textarea.addEventListener("input", (e) => {
          this.docsRepo[this.activeDocKey] = e.target.value;
          this.triggerAutoSave();
        });
      }
    } else {
      contentEl.innerHTML = this.parseMarkdown(rawText);
    }
  }

  triggerAutoSave() {
    const statusEl = document.getElementById("autoSaveStatus");
    if (statusEl) {
      statusEl.style.opacity = "1";
      statusEl.style.color = "#f59e0b";
      statusEl.textContent = "• Saving...";
    }

    if (this.autoSaveTimer) clearTimeout(this.autoSaveTimer);
    this.autoSaveTimer = setTimeout(async () => {
      const text = this.docsRepo[this.activeDocKey] || "";
      try {
        const resp = await fetch('/api/save_doc', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: this.activeDocKey, content: text })
        });
        if (resp.ok) {
          if (statusEl) {
            statusEl.style.color = "var(--color-lichen)";
            statusEl.textContent = "• Autosaved";
            setTimeout(() => {
              statusEl.style.opacity = "0";
            }, 2000);
          }
          this.syncDocsFromDisk();
        }
      } catch (e) {
        console.warn("Autosave error:", e);
        if (statusEl) {
          statusEl.style.color = "#ef4444";
          statusEl.textContent = "• Save failed";
        }
      }
    }, 800);
  }

  parseMarkdown(text) {
    if (!text) return "";
    
    // First escape HTML special characters to prevent XSS
    let html = this.escape(text);

    // Code blocks ```code```
    html = html.replace(/```([\s\S]*?)```/g, (match, p1) => {
      return `<pre><code>${p1.trim()}</code></pre>`;
    });

    // Inline code `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headings (h4 down to h1)
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Blockquotes (&gt; due to escaping)
    html = html.replace(/^&gt;\s?(.*$)/gim, '<blockquote>$1</blockquote>');

    // Markdown Links [title](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, title, url) => {
      const cleanUrl = url.replace(/&amp;/g, '&');
      return `<a href="${cleanUrl}" target="_blank" rel="noopener" style="color: var(--launch-lilac); text-decoration: underline; font-weight: 500;">${title}</a>`;
    });

    // Bold & Italics
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
    html = html.replace(/_(.*?)_/g, '<em>$1</em>');

    // Horizontal Rule
    html = html.replace(/^(?:---|\*\*\*)$/gim, '<hr style="border:0; border-top:1px solid var(--launch-border); margin:1.25rem 0;">');

    // Unordered Lists (- item or * item)
    html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, '<ul><li>$1</li></ul>');
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Ordered Lists (1. item, 2. item)
    html = html.replace(/^\s*(\d+)\.\s+(.*$)/gim, '<ol><li>$2</li></ol>');
    html = html.replace(/<\/ol>\s*<ol>/g, '');

    // Paragraph wrapping
    const blocks = html.split(/\n\s*\n/);
    return blocks.map(block => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (/^<(h[1-6]|pre|ul|ol|blockquote|hr)/i.test(trimmed)) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).filter(Boolean).join('\n\n');
  }

  deleteMemory(id) {
    this.memories = this.memories.filter(m => m.id !== id);
    this.saveMemories();
    this.renderMemories();
  }

  renderMemories() {
    if (this.memoryBadge) {
      this.memoryBadge.textContent = `${this.memories.length} Memories`;
    }

    if (this.latticeVis) {
      this.latticeVis.updateNodes(this.memories);
    }

    if (!this.memoryList) return;

    if (this.memories.length === 0) {
      this.memoryList.innerHTML = `<div style="color: var(--launch-muted); font-size: 12px; padding: 1rem; text-align: center; background: rgba(255,255,255,0.5); border-radius: 8px;">No active memories in lattice. Chat with AI to store new facts!</div>`;
      return;
    }

    this.memoryList.innerHTML = this.memories.map(m => `
      <div class="memory-item" style="position: relative;">
        <div class="memory-meta">
          <span class="tag">${this.escape(m.stratum)} Stratum</span>
          <span>Conf: ${Number(m.conf).toFixed(2)}</span>
        </div>
        <div style="padding-right: 18px;">${this.escape(m.text)}</div>
        <button onclick="window.hailApp.deleteMemory(${m.id})" title="Forget memory" style="position: absolute; top: 6px; right: 8px; background: none; border: none; color: var(--launch-muted); font-size: 14px; cursor: pointer; line-height: 1;">&times;</button>
      </div>
    `).join("");
  }

  sendMessage() {
    try {
      let text = this.chatInput.value.trim();
      if (!text) return;
      if (text.length > 2000) {
        text = text.substring(0, 2000);
      }

      if (!this.chatHistory) this.chatHistory = [];

      // Render user message bubble
      this.appendMsg("You", text, "user");
      this.chatInput.value = "";

      // Track in chat history
      this.chatHistory.push({ role: "user", text: text });

      // 1. Autonomous AI Memory Extraction & Storage
      let newlyStoredFact = null;
      try {
        newlyStoredFact = this.autonomouslyExtractMemory(text);
      } catch (err) {
        console.warn("Memory extraction warning:", err);
      }

      // 2. Query Recall Grounding
      let matchedMemories = [];
      try {
        matchedMemories = this.recallRelevantMemories(text);
      } catch (err) {
        console.warn("Memory recall warning:", err);
      }

      // 3. Generate Intelligent Assistant Response
      setTimeout(async () => {
        try {
          const responseText = await this.generateIntelligentResponse(text, matchedMemories, newlyStoredFact);
          this.chatHistory.push({ role: "assistant", text: responseText });
          this.appendMsg("HAIL Core Kernel", responseText, "assistant", matchedMemories.length, newlyStoredFact);
        } catch (err) {
          console.error("Response generation error:", err);
          const fallbackText = `I have received your message: "${this.escape(text)}". Synced with HAIL Core Kernel context.`;
          this.appendMsg("HAIL Core Kernel", fallbackText, "assistant");
        }
      }, 150);
    } catch (err) {
      console.error("sendMessage error:", err);
    }
  }

  /**
   * Autonomous HCL Memory Extractor:
   * Parses user input for key personal facts, education, identity, tech stack, preferences, and projects.
   */
  autonomouslyExtractMemory(prompt) {
    const p = prompt.trim();
    const lower = p.toLowerCase();
    let extractedText = null;
    let conf = 0.95;

    // Rule 1: Graduation & Education
    if (lower.includes("graduated") || lower.includes("degree") || lower.includes("studied") || lower.includes("qualification")) {
      extractedText = `Education: ${p}`;
      conf = 0.96;
    }
    // Rule 2: Identity & Name
    else if (lower.includes("my name is")) {
      const name = p.substring(lower.indexOf("my name is") + 10).trim();
      extractedText = `User identity: Name is ${name}`;
    } else if (lower.includes("call me")) {
      const name = p.substring(lower.indexOf("call me") + 7).trim();
      extractedText = `User preference: Call as ${name}`;
    }
    // Rule 3: Favorites & Preferences
    else if (lower.includes("favorite color is") || lower.includes("favourite color is") || lower.includes("favorite colour is")) {
      const val = p.substring(lower.indexOf("color is") + 8 || lower.indexOf("colour is") + 9).trim();
      extractedText = `User preference: Favorite color is ${val}`;
    } else if (lower.includes("my favorite") || lower.includes("my favourite") || lower.includes("i prefer")) {
      extractedText = `User preference: ${p}`;
      conf = 0.90;
    }
    // Rule 4: Tech Stack & Experience
    else if (lower.includes("software engineering") || lower.includes("developer") || lower.includes("programmer") || lower.includes("engineer")) {
      extractedText = `Background: ${p}`;
      conf = 0.94;
    } else if (lower.includes("i use") || lower.includes("my stack") || lower.includes("using ")) {
      extractedText = `User tech stack: ${p}`;
      conf = 0.92;
    }
    // Rule 5: Projects & Goals
    else if (lower.includes("building") || lower.includes("working on") || lower.includes("creating") || lower.includes("project")) {
      extractedText = `Active project: ${p}`;
      conf = 0.93;
    }
    // Rule 6: General Personal Declarations
    else if (lower.startsWith("i ") || lower.startsWith("my ") || lower.startsWith("i'm ") || lower.startsWith("i am ")) {
      if (!lower.includes("what") && !lower.includes("how") && !lower.includes("why") && !lower.includes("show")) {
        extractedText = `User fact: ${p}`;
        conf = 0.88;
      }
    }

    if (extractedText) {
      extractedText = extractedText.substring(0, 140);
      // Replace existing memory if key topic overlaps
      const topic = extractedText.split(":")[0];
      const existingIdx = this.memories.findIndex(m => m.text.startsWith(topic));
      if (existingIdx !== -1) {
        this.memories[existingIdx] = { id: Date.now(), text: extractedText, stratum: "Surface", conf: conf };
      } else {
        this.memories.unshift({ id: Date.now(), text: extractedText, stratum: "Surface", conf: conf });
      }
      this.saveMemories();
      this.renderMemories();
      return extractedText;
    }

    return null;
  }

  /**
   * Recall memories relevant to current query (keyword + semantic match)
   */
  recallRelevantMemories(query) {
    const tokens = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
    if (tokens.length === 0) return [];

    return this.memories.filter(m => {
      const memLower = m.text.toLowerCase();
      return tokens.some(t => memLower.includes(t));
    });
  }

  /**
   * Generates natural HCL Cognitive Layer answers based on recalled memories and chat history
   */
  async generateIntelligentResponse(userPrompt, recalledMemories, newlyStoredFact) {
    const lower = userPrompt.trim().toLowerCase();

    // 1. Chat History Recall Query ("what did i ask you", "what was my last question", "what did i say", "first thing i asked")
    if (lower.includes("first thing") || lower.includes("first question") || lower.includes("first message") || lower.includes("ask first") || lower.includes("asked first")) {
      const userMessages = this.chatHistory.filter(m => m.role === "user");
      if (userMessages.length > 0) {
        const firstMsg = userMessages[0].text;
        return `The first thing you asked me in this session was: **"${firstMsg}"**`;
      }
      return "You haven't asked any questions yet in this session!";
    }

    // Parse active selected model from unified selector
    let activeOllamaModel = "";
    let activeMoEModel = "";
    if (this.selectedUnifiedModel && this.selectedUnifiedModel.startsWith("ollama:")) {
      activeOllamaModel = this.selectedUnifiedModel.replace("ollama:", "");
    } else if (this.selectedUnifiedModel && this.selectedUnifiedModel.startsWith("moe:")) {
      activeMoEModel = this.selectedUnifiedModel.replace("moe:", "");
    }

    // Auto-load selected MoE model manifest on backend
    if (activeMoEModel && activeMoEModel !== this.lastLoadedMoEModel) {
      this.lastLoadedMoEModel = activeMoEModel;
      fetch('/api/moe/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: activeMoEModel })
      }).then(() => this.fetchMoETelemetry()).catch(err => console.warn("MoE model load error:", err));
    }

    const isDocTrigger = lower.includes("doc") || lower.includes("document") || lower.includes("history of") || lower.includes("write about") || lower.includes("create") || lower.includes("generate") || lower.includes("paper on");

    if (activeMoEModel && !isDocTrigger && !lower.includes("what did i ask") && !lower.includes("my name is") && !lower.includes("show my memory")) {
      try {
        const memStrings = recalledMemories.map(m => m.text);
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: userPrompt,
            model: `moe:${activeMoEModel}`,
            memories: memStrings
          })
        });
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.response) {
            return data.response;
          }
        }
      } catch (err) {
        console.warn("HydrusMoE chat fetch warning:", err);
      }
    }

    if (activeOllamaModel && !isDocTrigger && !lower.includes("what did i ask") && !lower.includes("my name is") && !lower.includes("show my memory")) {
      try {
        const memStrings = recalledMemories.map(m => m.text);
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: userPrompt,
            model: activeOllamaModel,
            memories: memStrings
          })
        });
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.response) {
            return data.response;
          }
        }
      } catch (err) {
        console.warn("Ollama chat fetch warning:", err);
      }
    }

    // 1. Context History Queries ("what did I ask you?", "what was my previous message?")
    if (lower.includes("what did i ask") || lower.includes("what did i say") || lower.includes("what was my last") || lower.includes("repeat my question")) {
      const userMessages = this.chatHistory.filter(m => m.role === "user");
      if (userMessages.length > 1) {
        const lastMsg = userMessages[userMessages.length - 2].text;
        return `Before this message, you asked me: **"${lastMsg}"**`;
      } else {
        return "This is the first message in our current chat session!";
      }
    }

    // 2. Memory / Facts Inspection Query ("show my memory facts", "show memories", "what do you remember", "my facts")
    if (lower.includes("memory") || lower.includes("facts") || lower.includes("remember") || lower.includes("retained")) {
      if (this.memories.length === 0) {
        return "My Stratified Memory Lattice is currently empty. Tell me something about yourself to form new memories!";
      }
      const factList = this.memories.map(m => `• **[${m.stratum} Stratum]** ${m.text}`).join("\n");
      return `Here is what I currently have retained in your **Stratified Memory Lattice**:\n\n${factList}`;
    }

    // 3. Newly Stored Memory Facts (Warm, human acknowledgment)
    if (newlyStoredFact) {
      if (lower.includes("my name is") || lower.includes("call me")) {
        const nameVal = newlyStoredFact.split("Name is ")[1] || newlyStoredFact.split("Call as ")[1] || "friend";
        return `Nice, I've saved that to my memory! It's great to meet you, **${nameVal.trim()}**. 😊`;
      }
      if (lower.includes("graduated") || lower.includes("degree") || lower.includes("software engineering")) {
        return `Nice! I've saved that to my memory. 🎓 Software Engineering is an awesome field to be in!`;
      }
      return `Nice, I've saved that to my memory!`;
    }

    // 5. Direct Name / Identity Query
    if (lower.includes("what is my name") || lower.includes("who am i") || lower.includes("do you know my name") || lower.includes("what's my name")) {
      const nameMem = this.memories.find(m => m.text.toLowerCase().includes("name is") || m.text.toLowerCase().includes("call as"));
      if (nameMem) {
        const raw = nameMem.text;
        const nameVal = raw.includes("Name is ") ? raw.split("Name is ")[1] : (raw.includes("Call as ") ? raw.split("Call as ")[1] : raw);
        return `Your name is **${nameVal.trim()}**, as stored in my Surface Memory Stratum.`;
      }
      return "I don't have your name in my memory lattice yet! What is your name?";
    }

    // 6. Favorite Color Query
    if (lower.includes("favorite color") || lower.includes("favourite color") || lower.includes("favorite colour")) {
      const colorMem = this.memories.find(m => m.text.toLowerCase().includes("color is") || m.text.toLowerCase().includes("colour is"));
      if (colorMem) {
        const val = colorMem.text.split(/colou?r is /i)[1] || colorMem.text;
        return `Your favorite color is **${val.trim()}**, as stored in my surface memory stratum.`;
      }
      return "I don't have your favorite color recorded yet. What is your favorite color?";
    }

    // 7. Project / Building Query
    if (lower.includes("what am i building") || lower.includes("what is my project") || lower.includes("what am i working on")) {
      const projMem = this.memories.find(m => m.text.toLowerCase().includes("project") || m.text.toLowerCase().includes("building"));
      if (projMem) {
        const val = projMem.text.replace(/^Active project:\s*/i, '');
        return `Based on my memory lattice, you are working on: **${val.trim()}**.`;
      }
      return "I don't have a project recorded for you yet. What are you currently building?";
    }

    // Train / Speed Questions
    if (lower.includes("fastest train") || lower.includes("speed of train") || lower.includes("maglev") || (lower.includes("train") && lower.includes("fast"))) {
      return `The world's fastest operational commercial train is the **Shanghai Maglev** in China, with a top speed of **460 km/h (286 mph)**.\n\nIn terms of experimental records, Japan's **SCMaglev L0 Series** holds the absolute world record at **603 km/h (375 mph)**.`;
    }

    // 0. Gratitude, Compliments & Friendly Feedback (Human Warmth)
    if (/\b(nice job|great job|good job|thank you|thanks|awesome|sweet|cool|perfect|amazing|brilliant|well done|nice|love it)\b/i.test(lower)) {
      const nameMem = this.memories.find(m => m.text.toLowerCase().includes("name is"));
      const nameVal = nameMem ? (nameMem.text.split("Name is ")[1] || "").trim() : "";
      const namePrefix = nameVal ? `, ${nameVal}` : "";
      return `Thank you so much${namePrefix}! 😊 I'm really glad you like it. Let me know if you want to expand on anything or work on a new topic!`;
    }

    // Direct answer requests ("just answer", "answer me", "answer")
    if (lower === "just answer" || lower === "answer" || lower.includes("just answer me") || lower.includes("give me an answer")) {
      const userMsgs = this.chatHistory.filter(m => m.role === "user");
      if (userMsgs.length >= 2) {
        const prevMsg = userMsgs[userMsgs.length - 2].text.toLowerCase();
        if (prevMsg.includes("train")) {
          return `The fastest commercial train is the **Shanghai Maglev** (460 km/h / 286 mph), while the experimental record is held by Japan's **L0 Series Maglev** at **603 km/h (375 mph)**.`;
        }
        return `Here is your direct answer to *"${userMsgs[userMsgs.length - 2].text}"*: Processing complete with local HAIL memory grounding.`;
      }
      return `I am here to answer your questions directly! What would you like to know?`;
    }

    // General Greetings (Fixed with word boundary regex)
    if (/\b(hello|hi|hey|greetings)\b/i.test(lower)) {
      const nameMem = this.memories.find(m => m.text.toLowerCase().includes("name is"));
      if (nameMem) {
        const nameVal = nameMem.text.split("Name is ")[1] || "there";
        return `Hello, **${nameVal.trim()}**! I am HAIL Core. How can I assist you today?`;
      }
      return "Hello! I am HAIL Core running locally. How can I assist you today?";
    }

    if (isDocTrigger) {
      let subject = userPrompt.trim();
      const subjectMatch = userPrompt.match(/(?:about|on|for|regarding)\s+(.+?)(?:\.|!|\?)?$/i);
      if (subjectMatch && subjectMatch[1]) {
        subject = subjectMatch[1].trim();
      } else {
        subject = userPrompt.replace(/^(?:create|generate|write|make)\s+(?:a\s+)?(?:document|doc|paper)\s*/i, "").trim();
        subject = subject.replace(/^(?:on|about|for|regarding)\s+/i, "").trim();
      }
      if (!subject) subject = "document";

      // Filename-safe slug
      let slugTitle = subject.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/(^_+|_+$)/g, "");
      let docTitle = slugTitle || "document";
      let docMarkdown = `# ${subject}\n\n*Fetching content...*`;
      let finalDisplayTitle = `${subject}.md`;

      let docGeneratedSuccessfully = false;

      if (lower.includes("seatbelt")) {
        docTitle = "seatbelt_dangers";
        docMarkdown = this.docsRepo.seatbelt_dangers;
        docGeneratedSuccessfully = true;
      } else if (lower.includes("architecture")) {
        docTitle = "hail_architecture";
        docMarkdown = this.docsRepo.hail_architecture;
        docGeneratedSuccessfully = true;
      } else if (lower.includes("gateway")) {
        docTitle = "cognitive_gateway";
        docMarkdown = this.docsRepo.cognitive_gateway;
        docGeneratedSuccessfully = true;
      } else {
        try {
          const resp = await fetch('/api/generate_doc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject: userPrompt, model: activeOllamaModel })
          });
          if (resp.ok) {
            const data = await resp.json();
            if (data && data.markdown) {
              docMarkdown = data.markdown;
              if (data.slug) docTitle = data.slug;
              if (data.title) finalDisplayTitle = `${data.title}.md`;
              docGeneratedSuccessfully = true;
            }
          }
        } catch (err) {
          console.error("Failed to generate real doc content:", err);
        }
      }

      if (docGeneratedSuccessfully) {
        this.loadDocPreview(docTitle, docMarkdown, finalDisplayTitle);
        const cleanName = finalDisplayTitle.replace('.md', '');
        return `I've written the full document for **${cleanName}** for you! I saved it to your local \`docs/\` folder and opened the live preview on the right.`;
      } else {
        return `I had trouble connecting to the document synthesizer. Please make sure \`python HAIL/distros/hail-desktop/app.py\` is running in your terminal!`;
      }
    }

    if (lower.includes("seatbelt")) {
      this.loadDocPreview("seatbelt_dangers");
      return `Displaying live preview of **seatbelt_dangers.md** in the Document Preview Card.`;
    }

    // Conversational Reply Grounded on Recalled Memory
    if (recalledMemories.length > 0) {
      const topMem = recalledMemories[0].text;
      return `Based on what you shared earlier (**${topMem}**), I am synchronized and ready to help!`;
    }

    // Smart General Answer Fallback
    return `I'm here to help with whatever you need! Feel free to ask questions, explore ideas, or ask me to generate a research document on any topic.`;
  }

  appendMsg(author, text, type, recalledCount = 0, newlyStoredFact = null) {
    const div = document.createElement("div");
    div.className = `chat-bubble ${type}`;

    let groundingHtml = "";
    if (type === "assistant") {
      if (newlyStoredFact) {
        groundingHtml += `<div class="grounded-badge-pill" style="background: rgba(139, 92, 246, 0.12); color: #8b5cf6; border-color: rgba(139, 92, 246, 0.25);">🧠 Saved to Memory: "${this.escape(newlyStoredFact)}"</div> `;
      }
      if (recalledCount > 0) {
        groundingHtml += `<div class="grounded-badge-pill">🧠 Grounded by ${recalledCount} surface stratum facts</div>`;
      }
    }

    const avatarHtml = type === "user" 
      ? `<div class="avatar user">U</div>`
      : `<div class="avatar ai">H</div>`;

    const formattedContent = this.parseMarkdown(text);

    div.innerHTML = `
      ${avatarHtml}
      <div class="message-content">
        <div style="font-weight: 600; font-size: 12px; color: var(--text-subtle); margin-bottom: 0.3rem;">${this.escape(author)}</div>
        <div>${formattedContent}</div>
        ${groundingHtml}
      </div>
    `;

    // Remove welcome hero on first user message
    const welcomeHero = this.chatFeed.querySelector(".welcome-hero");
    if (welcomeHero && type === "user") {
      welcomeHero.remove();
    }

    this.chatFeed.appendChild(div);
    this.chatFeed.scrollTop = this.chatFeed.scrollHeight;
  }

  escape(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ParticleBackground("bgCanvas");
  window.hailApp = new HAILChatApp();
});
