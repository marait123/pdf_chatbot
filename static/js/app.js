document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    
    // Append all selected files
    for (let file of fileInput.files) {
        formData.append('files', file);
    }

    const uploadSpinner = document.getElementById('upload-spinner');
    uploadSpinner.style.display = 'block';

    try {
        const response = await fetch('/documents/', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            loadUploadedDocuments();
            fileInput.value = ''; // Clear the file input
        } else {
            const error = await response.json();
            alert(error.message || 'Failed to upload documents');
        }
    } catch (error) {
        alert('Error uploading documents');
    } finally {
        uploadSpinner.style.display = 'none';
    }
});

async function loadUploadedDocuments() {
    const response = await fetch('/documents/');
    const data = await response.json();
    const documentsList = document.getElementById('documents-list');
    documentsList.innerHTML = '';
    data.documents.forEach(doc => {
        const li = document.createElement('li');
        li.textContent = doc.title;
        li.classList.add('list-group-item');
        documentsList.appendChild(li);
    });
}

document.getElementById('chat-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value;
    chatInput.value = '';
    addMessage('user', message);
    addLoadingMessage();

    const response = await fetch('/chat/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: message })
    });

    removeLoadingMessage();

    if (response.ok) {
        const data = await response.json();
        addMessage('ai', data.answer);
    } else {
        alert('Failed to get response');
    }
});

function addMessage(sender, text) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = createMessageElement(text, sender);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addLoadingMessage() {
    const messagesDiv = document.getElementById('messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('loading');
    loadingDiv.id = 'loading-message';
    loadingDiv.innerHTML = '<div class="spinner-border" role="status"></div> Generating response...';
    messagesDiv.appendChild(loadingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeLoadingMessage() {
    const loadingDiv = document.getElementById('loading-message');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function createMessageElement(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = sender === 'ai' ? 'markdown-content' : 'user-content';
    
    if (sender === 'ai') {
        contentDiv.innerHTML = marked.parse(text);
    } else {
        contentDiv.textContent = text;
    }
    
    messageDiv.appendChild(contentDiv);
    return messageDiv;
}

async function loadMessages() {
    console.log("adding messages ya bro")
    const response = await fetch('/messages/');
    const data = await response.json();
    data.forEach(msg => {
        addMessage(msg.type, msg.content);
    });
}

// Initialize the page
loadUploadedDocuments();
loadMessages();
