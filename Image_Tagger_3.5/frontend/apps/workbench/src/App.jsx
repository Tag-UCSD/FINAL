import React, { useEffect, useState, useCallback } from 'react';
import { ApiClient, Button, Header } from '@shared';
import { AlertCircle, Zap, Keyboard, CheckCircle2, XCircle, HelpCircle, Menu, ChevronLeft, ChevronRight } from 'lucide-react';

const api = new ApiClient('/api/v1/workbench');

export default function WorkbenchApp() {
    const [image, setImage] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [streak, setStreak] = useState(0);
    const [lastAction, setLastAction] = useState(null);
    
    // UX: Collapsible Sidebar State
    const [isSidebarOpen, setSidebarOpen] = useState(true);
    const [isMobile, setIsMobile] = useState(window.innerWidth < 1024);

    // Handle Window Resize
    useEffect(() => {
        const handleResize = () => {
            const mobile = window.innerWidth < 1024;
            setIsMobile(mobile);
            if (mobile) setSidebarOpen(false); // Auto-close on mobile
            else setSidebarOpen(true); // Auto-open on desktop
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

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
            if (err && err.isMaintenance) {
                setError('__MAINTENANCE__:' + (err.message || 'System temporarily unavailable (503).'));
            } else {
                setError((err && err.message) ? err.message : 'Failed to load next image');
            }
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
        
        setTimeout(() => setLastAction(null), 500);

        try {
            await api.post('/validate', {
                image_id: currentId,
                attribute_key: "global.relevance", 
                value: value,
                duration_ms: 1200 
            });
            loadNextImage();
        } catch (err) {
            if (err && err.isMaintenance) {
                setError('__MAINTENANCE__:' + (err.message || 'System temporarily unavailable (503).'));
            } else {
                setError("Failed to save decision: " + ((err && err.message) ? err.message : 'Unknown error'));
            }
            setLoading(false);
            setStreak(0);
        }
    };

    const handleKeyDown = useCallback((event) => {
        if (event.key === '1') handleDecision(0.0);
        if (event.key === '2') handleDecision(1.0);
        if (event.key === 'b') setSidebarOpen(prev => !prev); // 'b' for sidebar
    }, [image]);

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);

    const isMaintenance = typeof error === 'string' && error.startsWith('__MAINTENANCE__:');
    const maintenanceMessage = isMaintenance ? error.replace('__MAINTENANCE__:', '') : null;

    return (
        <div className="relative flex flex-col h-screen bg-gray-50 overflow-hidden">
            <Header appName="Workbench" title="Tagger Station" />

            {isMaintenance && (
                <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 text-gray-100">
                    <div className="bg-gray-900/90 px-6 py-4 rounded-lg shadow-lg max-w-md text-center">
                        <div className="font-semibold mb-2 text-sm">System temporarily unavailable</div>
                        <div className="text-xs mb-3">{maintenanceMessage || "The backend is reporting a maintenance / outage condition (503). Please pause tagging and try again shortly."}</div>
                        <div className="text-[10px] text-gray-400">If this persists for more than a few minutes, contact your TA or lab lead.</div>
                    </div>
                </div>
            )}

            {/* Quick Help & Toolbar */}
            <div className="border-b border-blue-100 bg-blue-50 px-4 py-2 text-xs text-blue-900 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <HelpCircle size={16} className="flex-shrink-0" />
                    <span className="hidden sm:inline">
                        Press <span className="font-semibold">1</span> (NO) or <span className="font-semibold">2</span> (YES). 
                        <span className="font-semibold"> 'b'</span> toggles sidebar.
                    </span>
                    <span className="sm:hidden">1=NO, 2=YES</span>
                </div>
                {/* Mobile Menu Toggle */}
                <button 
                    onClick={() => setSidebarOpen(!isSidebarOpen)}
                    className="p-1 hover:bg-blue-100 rounded text-blue-700 lg:hidden"
                >
                    <Menu size={18} />
                </button>
            </div>

<div className="mx-4 my-2 p-2 rounded-md bg-blue-50 border border-blue-100 text-[11px] text-blue-900 space-y-1">
    <p className="font-semibold">Workbench: what you do here</p>
    <ul className="list-disc ml-4 space-y-0.5">
        <li>This screen is for taggers: make fast, careful YES/NO decisions for each image.</li>
        <li>Use the keyboard shortcuts (1 = NO, 2 = YES) to stay on the canvas and minimize mouse movement.</li>
        <li>The large image area shows the current target; the right sidebar shows context, queue progress, and any extra guidance.</li>
        <li>Each decision is logged as a validation so supervisors can monitor consistency and quality later.</li>
    </ul>
</div>
            
            <div className="flex-1 flex relative overflow-hidden">
                {/* Main Canvas */}
                <main className={`flex-1 relative bg-black flex items-center justify-center group transition-all duration-300 ease-in-out ${isSidebarOpen && !isMobile ? 'mr-0' : ''}`}>
                    {loading && !image && (
                        <div className="text-white/50 animate-pulse flex flex-col items-center gap-2">
                            <Zap size={32} />
                            <span>Fetching Task...</span>
                        </div>
                    )}
                    
                    {error && (
                        <div className="absolute inset-0 bg-gray-900 z-50 flex flex-col items-center justify-center text-red-400">
                            <AlertCircle size={48} />
                            <p className="mt-4 font-bold text-xl">{error}</p>
                            <Button onClick={loadNextImage} variant="secondary" className="mt-6">Retry</Button>
                        </div>
                    )}

                    {image && !loading && (
                        <img 
                            src={image.url} 
                            alt="Tagging Target" 
                            className="max-w-full max-h-full object-contain shadow-2xl"
                        />
                    )}

                    {/* Feedback Animation */}
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

                    <div className="absolute top-4 right-4 bg-black/50 backdrop-blur px-4 py-2 rounded-full text-white font-mono text-sm border border-white/20">
                        🔥 Streak: {streak}
                    </div>
                    
                    {/* Desktop Sidebar Toggle (Floating on Canvas edge) */}
                    <button 
                        onClick={() => setSidebarOpen(!isSidebarOpen)}
                        className="absolute top-1/2 right-0 transform -translate-y-1/2 translate-x-1/2 bg-white border border-gray-200 rounded-full p-1 shadow-md hover:bg-gray-50 z-20 hidden lg:flex"
                    >
                        {isSidebarOpen ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                    </button>
                </main>

                {/* Tool Sidebar - Responsive Drawer */}
                <aside 
                    className={`
                        fixed lg:relative inset-y-0 right-0 z-30
                        w-80 bg-white border-l border-gray-200 flex flex-col shadow-2xl lg:shadow-none
                        transition-transform duration-300 ease-in-out
                        ${isSidebarOpen ? 'translate-x-0' : 'translate-x-full lg:hidden'}
                    `}
                >
                    <div className="p-6 border-b border-gray-100 flex justify-between items-center">
                        <h2 className="font-bold text-gray-800 text-sm uppercase tracking-wider">Current Task</h2>
                        {isMobile && (
                            <button onClick={() => setSidebarOpen(false)}>
                                <XCircle size={20} className="text-gray-400" />
                            </button>
                        )}
                    </div>
                    
                    <div className="p-6 border-b border-gray-100">
                        <p className="text-lg font-medium text-gray-900">Is this image "Modern"?</p>
                    </div>
                    
                    <div className="flex-1 p-6 flex flex-col gap-4 overflow-y-auto">
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 text-blue-900 text-sm leading-relaxed">
                            <strong>Instructions:</strong> Look for clean lines, glass curtains, lack of ornament, and industrial materials.
                        </div>
                        {/* Future: Region List can go here */}
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
                
                {/* Mobile Backdrop */}
                {isMobile && isSidebarOpen && (
                    <div 
                        className="fixed inset-0 bg-black/50 z-20"
                        onClick={() => setSidebarOpen(false)}
                    />
                )}
            </div>
        </div>
    );
}
