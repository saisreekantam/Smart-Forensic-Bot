import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Upload as UploadIcon, FileText, AlertCircle, CheckCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Upload = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [caseDescription, setCaseDescription] = useState("");
  const { toast } = useToast();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setSelectedFiles(prev => [...prev, ...files]);
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      toast({
        title: "No files selected",
        description: "Please select at least one file to upload.",
        variant: "destructive",
      });
      return;
    }

    const selectedCaseId = localStorage.getItem('selectedCaseId');
    if (!selectedCaseId) {
      toast({
        title: "No case selected",
        description: "Please select a case first.",
        variant: "destructive",
      });
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('case_id', selectedCaseId);
        if (caseDescription) {
          formData.append('description', caseDescription);
        }

        const response = await fetch(`http://localhost:8000/cases/${selectedCaseId}/evidence/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Failed to upload ${file.name}`);
        }

        setUploadProgress(Math.round(((i + 1) / selectedFiles.length) * 100));
      }

      toast({
        title: "Upload successful",
        description: `Successfully uploaded ${selectedFiles.length} file(s).`,
      });

      // Clear form
      setSelectedFiles([]);
      setCaseDescription("");
      
    } catch (error) {
      console.error('Upload error:', error);
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "An error occurred during upload.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'csv':
      case 'xml':
      case 'json':
      case 'txt':
      case 'pdf':
        return <FileText className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const getFileTypeColor = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'csv':
        return 'bg-green-100 text-green-800';
      case 'xml':
        return 'bg-blue-100 text-blue-800';
      case 'json':
        return 'bg-purple-100 text-purple-800';
      case 'txt':
        return 'bg-gray-100 text-gray-800';
      case 'pdf':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-orange-100 text-orange-800';
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Upload Evidence Files</h1>
        <Badge variant="outline" className="text-sm">
          UFDR Compatible
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UploadIcon className="h-5 w-5" />
              File Upload
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="files">Select Evidence Files</Label>
              <Input
                id="files"
                type="file"
                multiple
                accept=".csv,.xml,.json,.txt,.pdf,.ufdr"
                onChange={handleFileSelect}
                disabled={uploading}
              />
              <p className="text-sm text-muted-foreground">
                Supported formats: CSV, XML, JSON, TXT, PDF, UFDR
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Case Description (Optional)</Label>
              <Textarea
                id="description"
                placeholder="Enter case description or notes..."
                value={caseDescription}
                onChange={(e) => setCaseDescription(e.target.value)}
                disabled={uploading}
              />
            </div>

            {uploading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Uploading...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} />
              </div>
            )}

            <Button 
              onClick={handleUpload} 
              disabled={uploading || selectedFiles.length === 0}
              className="w-full"
            >
              {uploading ? "Uploading..." : `Upload ${selectedFiles.length} File(s)`}
            </Button>
          </CardContent>
        </Card>

        {/* Selected Files Preview */}
        <Card>
          <CardHeader>
            <CardTitle>Selected Files ({selectedFiles.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {selectedFiles.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No files selected</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {selectedFiles.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 border rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      {getFileIcon(file.name)}
                      <div>
                        <p className="font-medium truncate max-w-48">
                          {file.name}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge className={getFileTypeColor(file.name)}>
                        {file.name.split('.').pop()?.toUpperCase()}
                      </Badge>
                      {!uploading && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(index)}
                        >
                          ×
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Guidelines Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5" />
            Upload Guidelines
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold text-green-700 mb-2 flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                Supported File Types
              </h4>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>• CSV files (call logs, messages, contacts)</li>
                <li>• XML files (UFDR exports, system data)</li>
                <li>• JSON files (structured evidence data)</li>
                <li>• TXT files (reports, notes, transcripts)</li>
                <li>• PDF files (reports, documents, extracts)</li>
                <li>• UFDR files (mobile forensic data)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-blue-700 mb-2">Best Practices</h4>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>• Ensure files are from verified sources</li>
                <li>• Include case description for context</li>
                <li>• Upload files in logical groups</li>
                <li>• Verify file integrity before upload</li>
                <li>• Use descriptive file names</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Upload;