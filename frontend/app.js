const API_URL = "https://your-buddy-2.onrender.com";

// Elements
const dropArea = document.getElementById("dropArea");
const fileUpload = document.getElementById("fileUpload");
const uploadStatus = document.getElementById("uploadStatus");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const selectedFilesList = document.getElementById("selectedFilesList");
const uploadFilesBtn = document.getElementById("uploadFilesBtn");
const toolsSection = document.getElementById("toolsSection");
const toolBtns = document.querySelectorAll(".tool-btn");
const currentModeBadge = document.getElementById("currentModeBadge");
const modeDescription = document.getElementById("modeDescription");

const chatHistory = document.getElementById("chatHistory");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

let currentMode = "qa";
let isProcessing = false;
let fileUploaded = false;
let selectedFiles = [];

// Mode Descriptions
const modeDescriptions = {
  qa: "Ask any question about your documents.",
  quiz: "Generate a 5-question quiz for self-assessment.",
  simplify: "Get simple explanations of complex topics.",
  agent: "Handle multi-step analytical tasks.",
};

const modeBadges = {
  qa: "Q&A Mode",
  quiz: "Quiz Mode",
  simplify: "Simplify Mode",
  agent: "Agent Mode",
};

// --- Setup Event Listeners ---

// File Upload Trigger
if (fileUpload) fileUpload.addEventListener("change", handleFileSelection);
if (uploadFilesBtn) uploadFilesBtn.addEventListener("click", uploadSelectedFiles);

// Drag & Drop
if (dropArea) {
  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
  });
  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
  });
  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      selectedFiles = Array.from(e.dataTransfer.files);
      if (fileUpload) fileUpload.value = "";
      renderSelectedFiles();
    }
  });
}

// Tool Selection
if (toolBtns.length) {
  toolBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!fileUploaded) return; // wait until file is uploaded

      toolBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      currentMode = btn.dataset.mode;
      if (currentModeBadge) currentModeBadge.textContent = modeBadges[currentMode];
      if (modeDescription) modeDescription.textContent = modeDescriptions[currentMode];

      // Context-aware placeholders
      if (currentMode === "qa") userInput.placeholder = "Ask a question...";
      if (currentMode === "quiz")
        userInput.placeholder = "e.g., 'Chapter 1' or 'Photosynthesis'";
      if (currentMode === "simplify")
        userInput.placeholder = "What should I simplify?";
      if (currentMode === "agent") userInput.placeholder = "Describe the task...";

      userInput.focus();
    });
  });
}

// Chat Submit
if (chatForm && userInput && sendBtn) {
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!userInput.value.trim() || isProcessing || !fileUploaded) return;
    await processUserQuery(userInput.value.trim());
  });

  // Enter to submit (Shift+Enter for new line)
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Resize textarea automatically
  userInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
    if (this.value.trim() === "") {
      this.style.height = "48px";
    }
  });
} else {
  console.warn("Chat form or input elements are missing in the DOM.");
}

// --- Functions ---

function handleFileSelection() {
  selectedFiles = Array.from((fileUpload && fileUpload.files) || []);
  renderSelectedFiles();
}

function renderSelectedFiles() {
  if (!selectedFiles.length) {
    if (selectedFilesList) selectedFilesList.classList.add("hidden");
    if (uploadFilesBtn) uploadFilesBtn.classList.add("hidden");
    return;
  }

  if (selectedFilesList) selectedFilesList.classList.remove("hidden");
  if (uploadFilesBtn) {
    uploadFilesBtn.classList.remove("hidden");
    uploadFilesBtn.disabled = false;
  }

  // Update file count
  const filesCountEl = document.getElementById("filesCount");
  if (filesCountEl) {
    filesCountEl.textContent = `${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''} selected`;
  }

  // Render individual files
  const filesContainer = selectedFilesList.querySelector('.files-header');
  let filesHtml = `<div class="files-header">
    <i class="fas fa-check-circle"></i>
    <span id="filesCount">${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''} selected</span>
  </div>`;

  selectedFiles.forEach((file, index) => {
    filesHtml += `
      <div class="selected-file-item">
        <span title="${file.name}">${escapeHtml(file.name)}</span>
        <button type="button" class="remove-file-btn" onclick="removeSelectedFile(${index})">Remove</button>
      </div>
    `;
  });

  selectedFilesList.innerHTML = filesHtml;
}

window.removeSelectedFile = function (index) {
  selectedFiles.splice(index, 1);
  if (!selectedFiles.length && fileUpload) {
    fileUpload.value = "";
  }
  renderSelectedFiles();
};

async function uploadSelectedFiles() {
  if (!selectedFiles.length) {
    alert("Please select at least one file to upload.");
    return;
  }

  // UI Updates
  if (dropArea) dropArea.classList.add("hidden");
  if (uploadStatus) uploadStatus.classList.remove("hidden");

  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));

  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok) {
      if (uploadStatus) uploadStatus.classList.add("hidden");
      selectedFiles = [];
      if (fileUpload) fileUpload.value = "";
      renderSelectedFiles();
      if (fileInfo) fileInfo.classList.remove("hidden");
      if (fileName) fileName.textContent = result.message;
      if (toolsSection) toolsSection.classList.remove("hidden");

      // Enable inputs
      if (userInput) userInput.disabled = false;
      if (sendBtn) sendBtn.disabled = false;
      fileUploaded = true;

      appendBotMessage(
        "Files processed successfully! The knowledge base is ready. What would you like to do?",
      );
    } else {
      throw new Error(result.detail || "Upload failed");
    }
  } catch (error) {
    console.error(error);
    alert(`Error uploading files: ${error.message}`);
    if (uploadStatus) uploadStatus.classList.add("hidden");
    if (dropArea) dropArea.classList.remove("hidden");
  }
}

async function processUserQuery(query) {
  appendUserMessage(query);
  userInput.value = "";
  userInput.style.height = "48px";
  isProcessing = true;
  sendBtn.disabled = true;

  const loadingId = appendLoadingMessage();

  try {
    const response = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        mode: currentMode,
      }),
    });

    const result = await response.json();
    removeLoadingMessage(loadingId);

    if (response.ok) {
      if (currentMode === "quiz") {
        try {
          let jsonStr = result.response.trim();
          if (jsonStr.startsWith("```json")) {
            jsonStr = jsonStr
              .replace(/```json/g, "")
              .replace(/```/g, "")
              .trim();
          }
          const quizData = JSON.parse(jsonStr);
          renderInteractiveQuiz(quizData);
        } catch (e) {
          console.error("Failed to parse quiz JSON", e);
          appendBotMessage("Here is the generated quiz:\n\n" + result.response); // Fallback
        }
      } else {
        appendBotMessage(result.response);
      }
    } else {
      appendBotMessage(`Error: ${result.detail || "Failed to process query"}`);
    }
  } catch (error) {
    removeLoadingMessage(loadingId);
    appendBotMessage(
      `Connection Error: Make sure the FastAPI backend is running.`,
    );
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

// --- Quiz UI Logic ---

function renderInteractiveQuiz(quizData) {
  const div = document.createElement("div");
  div.className = "message bot-message";

  let quizHtml = `
    <div class="avatar"><i class="fas fa-robot"></i></div>
    <div class="content quiz-container">
      <h3 style="margin-top: 0; margin-bottom: 1.5rem;"><i class="fas fa-list-check"></i> Interactive Quiz</h3>
      <div id="quizQuestions">
  `;

  quizData.forEach((q, qIndex) => {
    quizHtml += `
      <div class="quiz-question" data-answer="${q.answer_index}">
        <p><strong>Q${qIndex + 1}: ${escapeHtml(q.question)}</strong></p>
        <div class="quiz-options">
    `;
    q.options.forEach((opt, oIndex) => {
      quizHtml += `<button class="quiz-option" onclick="selectQuizOption(this, ${qIndex}, ${oIndex})"><span class="opt-label">${String.fromCharCode(65 + oIndex)}</span> ${escapeHtml(opt)}</button>`;
    });
    quizHtml += `</div></div>`;
  });

  quizHtml += `
      </div>
      <button class="quiz-submit-btn" onclick="submitQuiz(this, ${quizData.length})">Submit Answers</button>
      <div class="quiz-result hidden"></div>
    </div>
  `;

  div.innerHTML = quizHtml;
  chatHistory.appendChild(div);
  scrollToBottom();
}

window.selectQuizOption = function (btn, qIndex, oIndex) {
  const optionsContainer = btn.parentElement;
  Array.from(optionsContainer.children).forEach((child) =>
    child.classList.remove("selected"),
  );
  btn.classList.add("selected");
  optionsContainer.dataset.answered = oIndex;
};

window.submitQuiz = function (submitBtn, totalQuestions) {
  const container = submitBtn.parentElement;
  const questions = container.querySelectorAll(".quiz-question");
  let score = 0;
  let allAnswered = true;

  questions.forEach((q) => {
    const optionsContainer = q.querySelector(".quiz-options");
    if (optionsContainer.dataset.answered === undefined) {
      allAnswered = false;
    }
  });

  if (!allAnswered) {
    alert("Please select an answer for all questions before submitting.");
    return;
  }

  questions.forEach((q) => {
    const correctIndex = parseInt(q.dataset.answer);
    const optionsContainer = q.querySelector(".quiz-options");
    const options = optionsContainer.querySelectorAll(".quiz-option");
    const answeredIndex = parseInt(optionsContainer.dataset.answered);

    options.forEach((opt, idx) => {
      opt.disabled = true;
      if (idx === correctIndex) {
        opt.classList.add("correct");
      } else if (idx === answeredIndex && answeredIndex !== correctIndex) {
        opt.classList.add("wrong");
      }
    });

    if (answeredIndex === correctIndex) {
      score++;
    }
  });

  submitBtn.classList.add("hidden");
  const resultDiv = container.querySelector(".quiz-result");
  resultDiv.classList.remove("hidden");
  const scoreColor = score === totalQuestions ? "#10b981" : "#a855f7";
  resultDiv.innerHTML = `<h4 style="color: ${scoreColor}; font-size: 1.2rem; margin: 0;">You scored ${score} out of ${totalQuestions}!</h4>`;
};

// --- UI Helpers ---

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user-message";
  div.innerHTML = `
    <div class="avatar"><i class="fas fa-user"></i></div>
    <div class="content"><p>${escapeHtml(text)}</p></div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
}

function appendBotMessage(text) {
  const div = document.createElement("div");
  div.className = "message bot-message";

  // Parse markdown (we included marked.js in html)
  let formattedText = `<p>${escapeHtml(text)}</p>`;
  if (typeof marked !== "undefined") {
    try {
      if (typeof marked.parse === "function") {
        formattedText = marked.parse(text);
      } else if (typeof marked === "function") {
        formattedText = marked(text);
      }
    } catch (error) {
      console.error("Markdown parse failed:", error);
      formattedText = `<p>${escapeHtml(text)}</p>`;
    }
  }

  div.innerHTML = `
    <div class="avatar"><i class="fas fa-robot"></i></div>
    <div class="content">${formattedText}</div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
}

function appendLoadingMessage() {
  const id = "loading-" + Date.now();
  const div = document.createElement("div");
  div.className = "message bot-message";
  div.id = id;
  div.innerHTML = `
    <div class="avatar"><i class="fas fa-robot"></i></div>
    <div class="content loader">
      <div class="dot"></div>
      <div class="dot"></div>
      <div class="dot"></div>
    </div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
  return id;
}

function removeLoadingMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
