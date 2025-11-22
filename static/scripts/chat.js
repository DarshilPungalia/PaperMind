const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const typingIndicator = document.getElementById('typingIndicator');
let fileListRefreshInterval;

// Configure marked.js
marked.setOptions({
    breaks: true,
    gfm: true
});

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Handle Enter key
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });

        // Handle form submission
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message to chat
            addMessage(message, 'user');
            
            // Clear input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Disable input
            messageInput.disabled = true;
            sendButton.disabled = true;
            
            // Show typing indicator
            showTypingIndicator();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage(data.error, 'error');
                } else {
                    addMessage(data.response, 'assistant', data.sources);
                }
                
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, there was an error processing your request.', 'error');
            } finally {
                // Hide typing indicator
                hideTypingIndicator();
                
                // Re-enable input
                messageInput.disabled = false;
                sendButton.disabled = false;
                messageInput.focus();
            }
        });

function addMessage(content, role, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    if (isMarkdownContent(content)) {
        messageDiv.innerHTML = marked.parse(content);
    } else {
        messageDiv.textContent = content;
    }
    
    // Add sources if available and role is assistant
    if (role === 'assistant' && sources && sources.length > 0) {
        const sourcesElement = createSourcesElement(sources);
        messageDiv.innerHTML += sourcesElement;
    }
    
    chatMessages.insertBefore(messageDiv, typingIndicator);
    scrollToBottom();
}

function showTypingIndicator() {
    typingIndicator.style.display = 'flex';
    scrollToBottom();
}

function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
}

        function scrollToBottom() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

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

function createSourcesElement(sources) {
    if (!sources || sources.length === 0) return '';
    
    const uniqueSources = Array.from(new Map(sources.map(s => [s.id, s])).values());
    
    const sourcesHtml = `
        <div class="sources-container">
            <div class="sources-header">Sources:</div>
            <div class="sources-list">
                ${uniqueSources.map(source => `
                    <div class="source-item">
                        <span class="source-icon">📄</span>
                        <span class="source-name">${escapeHtml(source.name)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    return sourcesHtml;
}
    
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

function startFileListRefresh() {
    fileListRefreshInterval = setInterval(fetchUploadMeta, 300000);
}

function stopFileListRefresh() {
    if (fileListRefreshInterval) {
        clearInterval(fileListRefreshInterval);
    }
}

// Focus on input when page loads
window.onload = function() {
    messageInput.focus();
    scrollToBottom();
    
    // Initialize file list
    fetchUploadMeta();
    startFileListRefresh();
};