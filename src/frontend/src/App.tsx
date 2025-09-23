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
import Upload from "./pages/Upload";
import Reports from "./pages/Reports";
import Analytics from "./pages/Analytics";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import CaseSelection from "./pages/CaseSelection";
import EvidenceViewer from "./pages/EvidenceViewer";
import NetworkAnalysis from "./pages/NetworkAnalysis";
import AIInvestigation from "./pages/AIInvestigation";

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
          
          {/* Case Selection - With Sidebar */}
          <Route path="/" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <CaseSelection />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          {/* Main App Routes - With Sidebar */}
          <Route path="/dashboard" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <Dashboard />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/query" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <ForensicChatbot />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/upload" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <Upload />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/reports" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <Reports />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/analytics" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <Analytics />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/evidence" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <EvidenceViewer />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/network" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <NetworkAnalysis />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/ai-investigation" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <AIInvestigation />
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/database" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <div className="text-center py-20">
                      <h2 className="text-2xl font-semibold">Database Search</h2>
                      <p className="text-muted-foreground mt-2">Database queries coming soon...</p>
                    </div>
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/users" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <div className="text-center py-20">
                      <h2 className="text-2xl font-semibold">User Management</h2>
                      <p className="text-muted-foreground mt-2">User administration coming soon...</p>
                    </div>
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="/settings" element={
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <SidebarInset>
                  <Header />
                  <main className="flex-1 p-6">
                    <div className="text-center py-20">
                      <h2 className="text-2xl font-semibold">System Settings</h2>
                      <p className="text-muted-foreground mt-2">System configuration coming soon...</p>
                    </div>
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          } />
          
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;