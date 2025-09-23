import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Plus, Search, FileText, Calendar, User, Building, AlertTriangle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useNavigate } from 'react-router-dom';

interface Case {
  id: string;
  case_number: string;
  title: string;
  status: string;
  investigator_name: string;
  created_at: string;
  updated_at: string;
  total_evidence_count: number;
  processed_evidence_count: number;
  processing_progress: number;
  description?: string;
  priority?: string;
  due_date?: string;
}

interface NewCase {
  case_number: string;
  title: string;
  investigator_name: string;
  description: string;
  department: string;
  priority: string;
  case_type: string;
  jurisdiction: string;
}

const CaseSelection = () => {
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newCase, setNewCase] = useState<NewCase>({
    case_number: '',
    title: '',
    investigator_name: '',
    description: '',
    department: '',
    priority: 'medium',
    case_type: '',
    jurisdiction: ''
  });

  const { toast } = useToast();
  const navigate = useNavigate();

  useEffect(() => {
    fetchCases();
  }, []);

  const fetchCases = async () => {
    try {
      const response = await fetch('http://localhost:8000/cases');
      if (response.ok) {
        const data = await response.json();
        setCases(data);
      } else {
        toast({
          title: "Error",
          description: "Failed to load cases",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('Error fetching cases:', error);
      toast({
        title: "Error",
        description: "Failed to connect to backend",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateCase = async () => {
    if (!newCase.case_number || !newCase.title || !newCase.investigator_name) {
      toast({
        title: "Error",
        description: "Please fill in all required fields",
        variant: "destructive"
      });
      return;
    }

    setIsCreating(true);
    try {
      const response = await fetch('http://localhost:8000/cases', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newCase),
      });

      if (response.ok) {
        const createdCase = await response.json();
        toast({
          title: "Success",
          description: "Case created successfully"
        });
        setIsCreateDialogOpen(false);
        // Store selected case in localStorage and navigate
        localStorage.setItem('selectedCaseId', createdCase.id);
        localStorage.setItem('selectedCaseNumber', createdCase.case_number);
        navigate('/dashboard');
      } else {
        const error = await response.json();
        toast({
          title: "Error",
          description: error.detail || "Failed to create case",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('Error creating case:', error);
      toast({
        title: "Error",
        description: "Failed to connect to backend",
        variant: "destructive"
      });
    } finally {
      setIsCreating(false);
    }
  };

  const handleSelectCase = (caseItem: Case) => {
    // Store selected case in localStorage
    localStorage.setItem('selectedCaseId', caseItem.id);
    localStorage.setItem('selectedCaseNumber', caseItem.case_number);
    navigate('/dashboard');
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'default';
      case 'pending':
        return 'secondary';
      case 'completed':
        return 'outline';
      case 'closed':
        return 'destructive';
      default:
        return 'secondary';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority?.toLowerCase()) {
      case 'critical':
        return 'text-red-600';
      case 'high':
        return 'text-orange-600';
      case 'medium':
        return 'text-yellow-600';
      case 'low':
        return 'text-green-600';
      default:
        return 'text-gray-600';
    }
  };

  const filteredCases = cases.filter(caseItem =>
    caseItem.case_number.toLowerCase().includes(searchQuery.toLowerCase()) ||
    caseItem.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    caseItem.investigator_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-100 mb-4">
            ForensicAI Investigation Platform
          </h1>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto">
            Select an existing case to continue your investigation, or create a new case to begin digital forensic analysis.
          </p>
        </div>

        {/* Search and Create */}
        <div className="flex flex-col sm:flex-row gap-4 mb-8">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
            <Input
              placeholder="Search cases by number, title, or investigator..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-slate-800 border-slate-600 text-slate-100 placeholder-slate-400 focus:border-blue-500"
            />
          </div>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button className="whitespace-nowrap bg-blue-600 hover:bg-blue-700 text-white">
                <Plus className="w-4 h-4 mr-2" />
                Create New Case
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl bg-slate-800 border-slate-700">
              <DialogHeader>
                <DialogTitle className="text-slate-100">Create New Investigation Case</DialogTitle>
                <DialogDescription className="text-slate-300">
                  Enter the details for the new forensic investigation case.
                </DialogDescription>
              </DialogHeader>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="case_number" className="text-slate-200">Case Number *</Label>
                  <Input
                    id="case_number"
                    placeholder="CASE-2024-001"
                    value={newCase.case_number}
                    onChange={(e) => setNewCase({ ...newCase, case_number: e.target.value })}
                    className="bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="investigator_name" className="text-slate-200">Lead Investigator *</Label>
                  <Input
                    id="investigator_name"
                    placeholder="Detective Smith"
                    value={newCase.investigator_name}
                    onChange={(e) => setNewCase({ ...newCase, investigator_name: e.target.value })}
                    className="bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400"
                  />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="title" className="text-slate-200">Case Title *</Label>
                  <Input
                    id="title"
                    placeholder="Digital Evidence Analysis - Mobile Device"
                    value={newCase.title}
                    onChange={(e) => setNewCase({ ...newCase, title: e.target.value })}
                    className="bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="department" className="text-slate-200">Department</Label>
                  <Input
                    id="department"
                    placeholder="Cybercrime Unit"
                    value={newCase.department}
                    onChange={(e) => setNewCase({ ...newCase, department: e.target.value })}
                    className="bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="priority" className="text-slate-200">Priority</Label>
                  <Select value={newCase.priority} onValueChange={(value) => setNewCase({ ...newCase, priority: value })}>
                    <SelectTrigger className="bg-slate-700 border-slate-600 text-slate-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-700 border-slate-600">
                      <SelectItem value="low" className="text-slate-100 hover:bg-slate-600">Low</SelectItem>
                      <SelectItem value="medium" className="text-slate-100 hover:bg-slate-600">Medium</SelectItem>
                      <SelectItem value="high" className="text-slate-100 hover:bg-slate-600">High</SelectItem>
                      <SelectItem value="critical" className="text-slate-100 hover:bg-slate-600">Critical</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="case_type" className="text-slate-200">Case Type</Label>
                  <Input
                    id="case_type"
                    placeholder="Mobile Forensics"
                    value={newCase.case_type}
                    onChange={(e) => setNewCase({ ...newCase, case_type: e.target.value })}
                    className="bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="jurisdiction" className="text-slate-200">Jurisdiction</Label>
                  <Input
                    id="jurisdiction"
                    placeholder="State Police"
                    value={newCase.jurisdiction}
                    onChange={(e) => setNewCase({ ...newCase, jurisdiction: e.target.value })}
                    className="bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400"
                  />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="description" className="text-slate-200">Description</Label>
                  <Textarea
                    id="description"
                    placeholder="Brief description of the case and investigation objectives..."
                    value={newCase.description}
                    onChange={(e) => setNewCase({ ...newCase, description: e.target.value })}
                    className="bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-6">
                <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)} className="border-slate-600 text-slate-200 hover:bg-slate-700">
                  Cancel
                </Button>
                <Button onClick={handleCreateCase} disabled={isCreating} className="bg-blue-600 hover:bg-blue-700 text-white">
                  {isCreating ? "Creating..." : "Create Case"}
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Cases Grid */}
        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto"></div>
            <p className="mt-4 text-slate-300">Loading cases...</p>
          </div>
        ) : filteredCases.length === 0 ? (
          <Card className="text-center py-12 bg-slate-800 border-slate-700">
            <CardContent>
              <FileText className="h-12 w-12 text-slate-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-100 mb-2">No Cases Found</h3>
              <p className="text-slate-300 mb-4">
                {searchQuery ? "No cases match your search criteria." : "Create your first investigation case to get started."}
              </p>
              {!searchQuery && (
                <Button onClick={() => setIsCreateDialogOpen(true)} className="bg-blue-600 hover:bg-blue-700 text-white">
                  <Plus className="w-4 h-4 mr-2" />
                  Create First Case
                </Button>
              )}
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredCases.map((caseItem) => (
              <Card 
                key={caseItem.id} 
                className="cursor-pointer hover:shadow-xl hover:shadow-blue-500/20 transition-all duration-200 border-l-4 border-l-blue-500 bg-slate-800 border-slate-700 hover:bg-slate-750"
                onClick={() => handleSelectCase(caseItem)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="text-lg text-slate-100 mb-1">
                        {caseItem.case_number}
                      </CardTitle>
                      <CardDescription className="text-sm font-medium text-slate-300">
                        {caseItem.title}
                      </CardDescription>
                    </div>
                    <Badge variant={getStatusBadgeVariant(caseItem.status)} className="bg-blue-600/20 text-blue-300 border-blue-500">
                      {caseItem.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center text-sm text-slate-300">
                      <User className="w-4 h-4 mr-2 text-blue-400" />
                      {caseItem.investigator_name}
                    </div>
                    
                    <div className="flex items-center text-sm text-slate-300">
                      <Calendar className="w-4 h-4 mr-2 text-blue-400" />
                      Created {new Date(caseItem.created_at).toLocaleDateString()}
                    </div>

                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">Evidence Files:</span>
                      <span className="font-medium text-slate-100">
                        {caseItem.processed_evidence_count}/{caseItem.total_evidence_count}
                      </span>
                    </div>

                    {caseItem.total_evidence_count > 0 && (
                      <div className="w-full bg-slate-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${caseItem.processing_progress}%` }}
                        />
                      </div>
                    )}

                    {caseItem.processing_progress < 100 && caseItem.total_evidence_count > 0 && (
                      <div className="flex items-center text-xs text-amber-400">
                        <AlertTriangle className="w-3 h-3 mr-1" />
                        Processing in progress ({caseItem.processing_progress.toFixed(0)}%)
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default CaseSelection;