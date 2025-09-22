import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { Header } from "@/components/layout/Header";
import Dashboard from "./pages/Dashboard";
import ForensicChatbot from "./pages/Chatbot";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Login Route - No Sidebar */}
          <Route path="/login" element={<Login />} />
          
          {/* Main App Routes - With Sidebar */}
          <Route path="/*" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/query" element={<ForensicChatbot />} />
                      <Route path="/upload" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">Upload UFDR</h2><p className="text-muted-foreground mt-2">File upload interface coming soon...</p></div>} />
                      <Route path="/reports" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">Reports</h2><p className="text-muted-foreground mt-2">Investigation reports coming soon...</p></div>} />
                      <Route path="/analytics" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">Analytics</h2><p className="text-muted-foreground mt-2">Data visualization coming soon...</p></div>} />
                      <Route path="/evidence" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">Evidence Viewer</h2><p className="text-muted-foreground mt-2">Evidence browser coming soon...</p></div>} />
                      <Route path="/network" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">Network Analysis</h2><p className="text-muted-foreground mt-2">Connection mapping coming soon...</p></div>} />
                      <Route path="/database" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">Database Search</h2><p className="text-muted-foreground mt-2">Database queries coming soon...</p></div>} />
                      <Route path="/users" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">User Management</h2><p className="text-muted-foreground mt-2">User administration coming soon...</p></div>} />
                      <Route path="/settings" element={<div className="text-center py-20"><h2 className="text-2xl font-semibold">System Settings</h2><p className="text-muted-foreground mt-2">System configuration coming soon...</p></div>} />
                      <Route path="*" element={<NotFound />} />
                    </Routes>
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;