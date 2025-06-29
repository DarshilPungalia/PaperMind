let inputCounter = 0;
let isUploading = false;
const inputSummary = document.getElementById("inputSummary");
const docForm = document.getElementById("docForm");
const loader = document.getElementById("loader");
const submitBtn = document.getElementById("submitBtn");
const resultDiv = document.getElementById("result");

// Configure marked.js for better rendering
marked.setOptions({
    breaks: true,
    gfm: true
});

// Enhanced markdown detection
function isMarkdownContent(content) {
    if (typeof content !== 'string' || !content.trim()) return false;
            
    const markdownIndicators = [
        /^#{1,6}\s+/m,           // Headers
        /\*\*.*?\*\*/,           // Bold
        /\*.*?\*/,               // Italic
        /^[-*+]\s+/m,            // Unordered lists
        /^\d+\.\s+/m,            // Ordered lists
        /`[^`]+`/,               // Inline code
        /^\>\s+/m,               // Blockquotes
        /\[.*?\]\(.*?\)/,        // Links
        /\|.*?\|/,               // Tables
        /^---+$/m                // Horizontal rules
    ];
            
    return markdownIndicators.some(pattern => pattern.test(content));
}

function addInputGroup() {
    inputCounter++;
    const container = document.getElementById("inputContainer");
            
    const inputGroup = document.createElement("div");
    inputGroup.className = "input-group";
    inputGroup.id = `input-group-${inputCounter}`;
            
    inputGroup.innerHTML = `
        <div class="input-group-header">
            <span class="input-group-title">Input Source</span>
                    ${inputCounter > 1 ? `<button type="button" class="remove-input" onclick="removeInputGroup(${inputCounter})">Remove</button>` : ''}
        </div>
                
        <label for="file_type_${inputCounter}">File Type:</label>
        <select name="file_type_${inputCounter}" id="file_type_${inputCounter}" required>
            <option value="text">Text File</option>
            <option value="pdf">PDF File</option>
            <option value="code">Code File</option>
            <option value="link">Web URL</option>
            <option value="pasted">Pasted Text</option>
        </select>

        <div id="file_input_container_${inputCounter}">
            <label for="file_${inputCounter}">Upload File:</label>
            <input type="file" name="file_${inputCounter}" id="file_${inputCounter}">
            <small id="fileHint_${inputCounter}" style="color: #bbb;"></small>
        </div>

        <div id="url_input_container_${inputCounter}" style="display: none;">
            <label for="url_${inputCounter}">Enter URL:</label>
            <input type="text" name="url_${inputCounter}" id="url_${inputCounter}">
        </div>

        <div id="paste_input_container_${inputCounter}" style="display: none;">
            <label for="pasted_${inputCounter}">Paste Text:</label>
            <textarea name="pasted_${inputCounter}" id="pasted_${inputCounter}" rows="8"></textarea>
        </div>
    `;
            
    container.appendChild(inputGroup);
            
    // Add event listener for the new file type select
    const fileTypeSelect = document.getElementById(`file_type_${inputCounter}`);
    fileTypeSelect.addEventListener("change", function() {
        updateInputVisibility(inputCounter, this.value);
    });
            
    // Initialize visibility
    updateInputVisibility(inputCounter, "text");
    updateInputSummary();
}

function removeInputGroup(id) {
    const inputGroup = document.getElementById(`input-group-${id}`);
    if (inputGroup) {
        inputGroup.remove();
        updateInputSummary();
        // Trigger auto-upload after removal
        debounceAutoUpload();
    }
}

function updateInputVisibility(id, fileType) {
    const fileInputContainer = document.getElementById(`file_input_container_${id}`);
    const urlInputContainer = document.getElementById(`url_input_container_${id}`);
    const pasteInputContainer = document.getElementById(`paste_input_container_${id}`);
    const fileHint = document.getElementById(`fileHint_${id}`);
    const fileInput = document.getElementById(`file_${id}`);

    fileInputContainer.style.display = fileType === "text" || fileType === "pdf" || fileType === "code" ? "block" : "none";
    urlInputContainer.style.display = fileType === "link" ? "block" : "none";
    pasteInputContainer.style.display = fileType === "pasted" ? "block" : "none";

    if (fileType === "text") {
        fileInput.setAttribute("accept", ".txt");
        fileHint.textContent = "Allowed: .txt files only";
    } else if (fileType === "pdf") {
        fileInput.setAttribute("accept", ".pdf");
        fileHint.textContent = "Allowed: .pdf files only";
    } else if (fileType === "code") {
        const codeExtensions = [
            ".cpp", ".cc", ".cxx", ".hpp", ".h",
            ".go", ".java", ".kt", ".kts",
            ".js", ".mjs", ".cjs",
            ".ts", ".tsx",
            ".php", ".phtml", ".php3", ".php4",
            ".proto", ".py", ".pyw", ".rst",
            ".rb", ".erb", ".rs",
            ".scala", ".sc", ".swift",
            ".md", ".markdown",
            ".tex", ".ltx", ".latex",
            ".html", ".htm", ".sol",
            ".cs", ".ipynb",
            ".cob", ".cbl", ".cpy",
            ".c", ".lua",
            ".pl", ".pm", ".t", ".pod",
            ".hs", ".lhs",
            ".ex", ".exs",
            ".ps1", ".psm1", ".psd1"
        ];
        fileInput.setAttribute("accept", codeExtensions.join(","));
        fileHint.textContent = "Allowed: Programming files (.py, .js, .cpp, .java, .html, .cs, etc.)";
    } else {
        fileInput.removeAttribute("accept");
        fileHint.textContent = "";
    }
}

function updateInputSummary() {
    const inputGroups = document.querySelectorAll('.input-group');
    const count = inputGroups.length;
    inputSummary.innerHTML = `<span class="input-count">${count}</span> input source(s) configured`;
}

// Add upload status functionality
function showUploadStatus(message, type = 'loading') {
    const statusDiv = document.getElementById('uploadStatus') || createUploadStatus();
    statusDiv.textContent = message;
    statusDiv.className = `upload-status ${type}`;
    statusDiv.style.display = 'block';
            
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
    }
}

function createUploadStatus() {
    const statusDiv = document.createElement('div');
    statusDiv.id = 'uploadStatus';
    statusDiv.className = 'upload-status';
    document.querySelector('.input-summary').after(statusDiv);
    return statusDiv;
}

// Debounce function to prevent too many rapid uploads
let uploadTimeout;
function debounceAutoUpload() {
    clearTimeout(uploadTimeout);
    uploadTimeout = setTimeout(autoUploadWithRefresh, 500); 
}

async function autoUploadWithRefresh() {
    if (isUploading) return;
    
    const inputGroups = document.querySelectorAll('.input-group');
    let hasValidInput = false;
    const formData = new FormData();
    let validInputCount = 0;
    
    // Check if there's any valid input and build FormData
    for (let group of inputGroups) {
        const groupId = group.id.split('-')[2];
        const fileTypeSelect = group.querySelector(`select[name="file_type_${groupId}"]`);
        const fileType = fileTypeSelect.value;
        
        let hasValidInputForThisGroup = false;
        
        if (fileType === "text" || fileType === "pdf" || fileType === "code") {
            const fileInput = group.querySelector(`input[name="file_${groupId}"]`);
            if (fileInput.files && fileInput.files.length > 0) {
                formData.append(`file_type_${groupId}`, fileType);
                formData.append(`file_${groupId}`, fileInput.files[0]);
                hasValidInputForThisGroup = true;
            }
        } else if (fileType === "link") {
            const urlInput = group.querySelector(`input[name="url_${groupId}"]`);
            if (urlInput.value.trim()) {
                formData.append(`file_type_${groupId}`, fileType);
                formData.append(`url_${groupId}`, urlInput.value);
                hasValidInputForThisGroup = true;
            }
        } else if (fileType === "pasted") {
            const pastedInput = group.querySelector(`textarea[name="pasted_${groupId}"]`);
            if (pastedInput.value.trim()) {
                formData.append(`file_type_${groupId}`, fileType);
                formData.append(`pasted_${groupId}`, pastedInput.value);
                hasValidInputForThisGroup = true;
            }
        }
        
        if (hasValidInputForThisGroup) {
            hasValidInput = true;
            validInputCount++;
        }
    }
    
    if (!hasValidInput) {
        console.log('No valid input found for auto-upload');
        return;
    }
    
    console.log(`Auto-uploading ${validInputCount} input(s)...`);
    
    isUploading = true;
    showUploadStatus(`Uploading and indexing document(s)...`, 'loading');
    
    let uploadSuccess = false;
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const message = await response.text();
            showUploadStatus(message, 'success');
            uploadSuccess = true;
            console.log('Auto-upload successful:', message);
        } else {
            const errorText = await response.text();
            showUploadStatus(`Upload failed: ${errorText}`, 'error');
            console.error('Upload failed:', errorText);
        }
        
    } catch (error) {
        showUploadStatus(`Upload error: ${error.message}`, 'error');
        console.error('Upload error:', error);
    } finally {
        isUploading = false;
        
        // Refresh file list after successful upload
        if (uploadSuccess) {
            setTimeout(fetchUploadMeta, 1000); // Wait 1 second then refresh
        }
    }
}

// Sidebar functionality
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const body = document.body;
    
    if (sidebar.classList.contains('open')) {
        closeSidebar();
    } else {
        sidebar.classList.add('open');
        overlay.classList.add('active');
        body.classList.add('sidebar-open');
        fetchUploadMeta(); // Refresh file list when opening
    }
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const body = document.body;
    
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
    body.classList.remove('sidebar-open');
}

// Fetch upload metadata from backend
async function fetchUploadMeta() {
    try {
        const response = await fetch('/upload-meta');
        const data = await response.json();
        updateFileList(data);
    } catch (error) {
        console.error('Error fetching upload metadata:', error);
        updateFileList({ count: 0, files: [] });
    }
}

// Update the file list in sidebar
function updateFileList(meta) {
    const fileList = document.getElementById('fileList');
    
    // Update file list
    if (meta.files && meta.files.length > 0) {
        fileList.innerHTML = meta.files.map(file => {
            const uploadTime = new Date(file.uploaded_at).toLocaleString();
            return `
        <div class="file-item">
            <div class="file-name">${escapeHtml(file.name)}</div>
        </div>
            `;
        }).join('');
    } else {
        fileList.innerHTML = '<div class="no-files">No files uploaded yet</div>';
    }
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-refresh file list periodically (optional)
let fileListRefreshInterval;

function startFileListRefresh() {
    // Refresh every 5 mins
    fileListRefreshInterval = setInterval(fetchUploadMeta, 300000);
}

function stopFileListRefresh() {
    if (fileListRefreshInterval) {
        clearInterval(fileListRefreshInterval);
    }
}

// Add event listeners for auto-upload
document.addEventListener("change", function(e) {
    const el = e.target;
    if (el.matches("input[type='file'], input[type='text'], textarea")) {
        debounceAutoUpload();
    }
});

// Handle main form submission
docForm.addEventListener("submit", function (e) {
    const inputGroups = document.querySelectorAll('.input-group');
    let hasValidInput = false;
            
    // Validate each input group
    for (let i = 0; i < inputGroups.length; i++) {
        const inputGroup = inputGroups[i];
        const groupId = inputGroup.id.split('-')[2];
        const fileTypeSelect = inputGroup.querySelector(`select[name="file_type_${groupId}"]`);
        const fileType = fileTypeSelect.value;
                
        if (fileType === "text" || fileType === "pdf" || fileType === "code") {
            const fileInput = inputGroup.querySelector(`input[name="file_${groupId}"]`);
            if (fileInput.files && fileInput.files.length > 0) {
                hasValidInput = true;
            }
        } else if (fileType === "link") {
            const urlInput = inputGroup.querySelector(`input[name="url_${groupId}"]`);
            if (urlInput.value.trim()) {
                hasValidInput = true;
            }
        } else if (fileType === "pasted") {
            const pastedInput = inputGroup.querySelector(`textarea[name="pasted_${groupId}"]`);
            if (pastedInput.value.trim()) {
                hasValidInput = true;
            }
        }
    }
            
    if (!hasValidInput) {
        alert("Please provide at least one valid input source.");
        e.preventDefault();
        return false;
    }
            
    loader.style.display = "block";
    submitBtn.disabled = true;
    submitBtn.textContent = "Processing...";
});
        
function renderMarkdown(content) {
    if (!isMarkdownContent(content)) {
        return content.replace(/\n/g, '<br>');
    }
            
    return marked.parse(content);
}
        
function applyTypingEffect() {
    const resultContent = resultDiv.innerHTML;
    if (!resultContent || resultContent.trim() === '') return;
            
    const rawContent = resultDiv.textContent || resultDiv.innerText || '';
            
    if (rawContent.trim()) {
        const renderedHtml = renderMarkdown(rawContent);
                
        if (renderedHtml !== rawContent) {
            resultDiv.innerHTML = '';
            let i = 0;
                    
            function typeNextChar() {
                if (i < renderedHtml.length) {
                    resultDiv.innerHTML = renderedHtml.substring(0, i + 1);
                    i++;
                    setTimeout(typeNextChar, 5);
                }
            }
                    
            typeNextChar();
        } else {
            resultDiv.innerHTML = '';
            let i = 0;
                    
            function typeNextChar() {
                if (i < resultContent.length) {
                    resultDiv.innerHTML += resultContent.charAt(i);
                    i++;
                    setTimeout(typeNextChar, 5);
                }
            }
                    
            typeNextChar();
        }
    }
}
        
// Function to manually convert markdown (useful for testing)
function convertToMarkdown() {
    const content = resultDiv.textContent || resultDiv.innerText || '';
    if (content.trim()) {
        const markdownHtml = renderMarkdown(content);
        resultDiv.innerHTML = markdownHtml;
    }
}
        
// Initialize with first input group
window.onload = function() {
    addInputGroup();

    // Set active navigation
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
    
    if (resultDiv.innerHTML && resultDiv.innerHTML.trim() !== '') {
        applyTypingEffect();
    }
    
    // Initialize file list
    fetchUploadMeta();
    startFileListRefresh();
};
        
console.log('To test markdown rendering, call convertToMarkdown() in the console');
