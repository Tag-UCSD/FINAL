/**
 * Centralized API Client.
 * Handles JWT injection and standardized error parsing.
 */
export class ApiClient {
  constructor(baseUrl = '/api', defaultHeaders = {}) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = defaultHeaders;
    }

    async get(endpoint) {
        return this._request(endpoint, { method: 'GET' });
    }

    async post(endpoint, body, options = {}) {
        return this._request(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this.defaultHeaders,
                ...(options.headers || {}),
            },
            body: JSON.stringify(body)
        });
    }

    async delete(endpoint) {
        return this._request(endpoint, { method: 'DELETE' });
    }

    async put(endpoint, body, options = {}) {
        return this._request(endpoint, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                ...this.defaultHeaders,
                ...(options.headers || {}),
            },
            body: JSON.stringify(body)
        });
    }

    async _request(endpoint, options) {
        // Mock Auth Header - In production this comes from AuthContext
        const headers = { 
            'X-User-ID': '1', 
            ...options.headers 
        };
        
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, { ...options, headers });
            
if (!response.ok) {
    let payload = {};
    try {
        payload = await response.json();
    } catch (e) {
        payload = {};
    }
    const message = payload.detail || payload.message || `API Request Failed: ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    if (response.status === 503) {
        // Hint to the UI that this is a maintenance / outage condition.
        error.isMaintenance = true;
        // Broadcast a global maintenance event so that top-level shells
        // (Explorer / Workbench / Admin / Monitor) can show a full-screen
        // overlay rather than a cryptic 'fetch failed' toast.
        if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
            const detail = {
                status: response.status,
                endpoint: `${this.baseUrl}${endpoint}`,
                message,
            };
            window.dispatchEvent(new CustomEvent('image-tagger:maintenance', { detail }));
        }
    }
    throw error;
}
            
            // Handle 204 No Content
            if (response.status === 204) return null;

            return await response.json();
        } catch (err) {
            console.error(`API Error [${endpoint}]:`, err);
            throw err;
        }
    }
}