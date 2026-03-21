import React, { useEffect, useState, useCallback } from 'react';
import { ApiClient, Button, Header } from '@shared';
import { AlertCircle, Zap, Keyboard, CheckCircle2, XCircle, HelpCircle } from 'lucide-react';

const api = new ApiClient('/api/v1/workbench');

export default function WorkbenchApp() {
    const [image, setImage] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [streak, setStreak] = useState(0);
    const [lastAction, setLastAction] = useState(null); // 'accept' or 'reject' for feedback animation

    useEffect(() => {
        loadNextImage();
    }, []);

    const loadNextImage = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.get('/next');
            setImage(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleDecision = async (value) => {
        if (!image) return;
        
        const currentId = image.id;
        const action = value === 1.0 ? 'accept' : 'reject';
        
        // 1. Optimistic UI Update
        setLastAction(action);
        setImage(null); 
        setLoading(true);
        setStreak(s => s + 1);

        // Clear animation trigger after short delay
        setTimeout(() => setLastAction(null), 500);

        try {
            // 2. Fire and Forget (in this simple version)
            await api.post('/validate', {
                image_id: currentId,
                attribute_key: "global.relevance", 
                value: value,
                duration_ms: 1200 
            });
            // 3. Fetch Next
            loadNextImage();
        } catch (err) {
            setError("Failed to save decision: " + err.message);
            setLoading(false);
            setStreak(0);
        }
    };

    const handleKeyDown = useCallback((event) => {
        if (event.key === '1') handleDecision(0.0);
        if (event.key === '2') handleDecision(1.0);
    }, [image]);

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            <Header appName="Workbench" title="Tagger Station" />

{/* Quick Help */}
<div className="border-b border-blue-100 bg-blue-50 px-4 py-2 text-xs text-blue-900 flex items-center gap-2">
    <HelpCircle size={16} className="flex-shrink-0" />
    <span>
        One binary question at a time. Press <span className="font-semibold">1</span> for NO,
        <span className="font-semibold"> 2</span> for YES, or use the mouse buttons. Use <span className="font-semibold">Retry</span> if something looks wrong.
    </span>
</div>
            
            <div className="flex-1 flex overflow-hidden">
                {/* Main Canvas */}
                <main className="flex-1 relative bg-black flex items-center justify-center group">
                    {/* Loading State */}
                    {loading && !image && (
                        <div className="text-white/50 animate-pulse flex flex-col items-center gap-2">
                            <Zap size={32} />
                            <span>Fetching Task...</span>
                        </div>
                    )}
                    
                    {/* Error State */}
                    {error && (
                        <div className="absolute inset-0 bg-gray-900 z-50 flex flex-col items-center justify-center text-red-400">
                            <AlertCircle size={48} />
                            <p className="mt-4 font-bold text-xl">{error}</p>
                            <Button onClick={loadNextImage} variant="secondary" className="mt-6">Retry</Button>
                        </div>
                    )}

                    {/* Active Image */}
                    {image && !loading && (
                        <img 
                            src={image.url} 
                            alt="Tagging Target" 
                            className="max-w-full max-h-full object-contain shadow-2xl transition-transform duration-200"
                        />
                    )}

                    {/* Feedback Animation Overlay */}
                    {lastAction === 'accept' && (
                        <div className="absolute inset-0 flex items-center justify-center bg-green-500/20 pointer-events-none animate-ping">
                            <CheckCircle2 size={128} className="text-green-400" />
                        </div>
                    )}
                    {lastAction === 'reject' && (
                        <div className="absolute inset-0 flex items-center justify-center bg-red-500/20 pointer-events-none animate-ping">
                            <XCircle size={128} className="text-red-400" />
                        </div>
                    )}

                    {/* Floating Stats */}
                    <div className="absolute top-4 right-4 bg-black/50 backdrop-blur px-4 py-2 rounded-full text-white font-mono text-sm border border-white/20">
                        🔥 Streak: {streak}
                    </div>
                </main>

                {/* Tool Sidebar */}
                <aside className="w-80 bg-white border-l border-gray-200 flex flex-col shadow-xl z-10">
                    <div className="p-6 border-b border-gray-100">
                        <h2 className="font-bold text-gray-800 text-sm uppercase tracking-wider">Current Task</h2>
                        <p className="text-lg font-medium text-gray-900 mt-2">Is this image "Modern"?</p>
                    </div>
                    
                    <div className="flex-1 p-6 flex flex-col gap-4">
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 text-blue-900 text-sm leading-relaxed">
                            <strong>Instructions:</strong> Look for clean lines, glass curtains, lack of ornament, and industrial materials.
                        </div>
                    </div>

                    <div className="p-6 border-t border-gray-200 bg-gray-50">
                        <div className="grid grid-cols-2 gap-4">
                            <Button onClick={() => handleDecision(0.0)} variant="danger" className="h-20 flex flex-col">
                                <span className="text-2xl font-bold">NO</span>
                                <span className="text-xs uppercase opacity-75 font-mono bg-black/10 px-2 py-1 rounded">Key: 1</span>
                            </Button>
                            <Button onClick={() => handleDecision(1.0)} variant="primary" className="h-20 flex flex-col">
                                <span className="text-2xl font-bold">YES</span>
                                <span className="text-xs uppercase opacity-75 font-mono bg-white/20 px-2 py-1 rounded">Key: 2</span>
                            </Button>
                        </div>
                        <div className="mt-4 flex items-center justify-center text-gray-400 text-xs gap-2">
                            <Keyboard size={14} />
                            <span>Shortcuts Active</span>
                        </div>
                    </div>
                </aside>
            </div>
        </div>
    );
}