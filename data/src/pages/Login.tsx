import { useState } from "react";
import { Eye, EyeOff, Shield, Lock, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import investigationIcon from "@/assets/investigation-icon.png";

const Login = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle login logic here
    console.log("Login attempt:", { email, password, rememberMe });
  };

  return (
    <div className="min-h-screen bg-gradient-subtle flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        {/* Logo & Branding */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-glow">
            <img src={investigationIcon} alt="ForensicAI" className="w-8 h-8" />
          </div>
          <h1 className="text-3xl font-bold mb-2">ForensicAI</h1>
          <p className="text-muted-foreground">Secure Digital Investigation Platform</p>
        </div>

        {/* Login Form */}
        <Card className="card-professional shadow-glow border-border/50">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl font-semibold text-center flex items-center justify-center gap-2">
              <Lock className="w-5 h-5 text-primary" />
              Secure Access
            </CardTitle>
            <CardDescription className="text-center">
              Enter your credentials to access the investigation platform
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Email Field */}
              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-medium">
                  Official Email Address
                </Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="detective@forensics.gov"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-10 bg-secondary/50 border-border/50 focus:bg-secondary/70"
                    required
                  />
                </div>
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <Label htmlFor="password" className="text-sm font-medium">
                  Password
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your secure password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 pr-10 bg-secondary/50 border-border/50 focus:bg-secondary/70"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Remember Me & Forgot Password */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="remember"
                    checked={rememberMe}
                    onCheckedChange={(checked) => setRememberMe(checked as boolean)}
                  />
                  <Label htmlFor="remember" className="text-sm text-muted-foreground">
                    Remember me
                  </Label>
                </div>
                <Button variant="link" className="p-0 h-auto text-primary hover:text-primary/80">
                  Forgot password?
                </Button>
              </div>

              {/* Login Button */}
              <Button 
                type="submit" 
                variant="professional" 
                size="lg" 
                className="w-full"
                disabled={!email || !password}
              >
                <Shield className="mr-2 w-4 h-4" />
                Secure Login
              </Button>
            </form>

            {/* Security Notice */}
            <div className="p-4 bg-muted/20 rounded-lg border border-border/30">
              <div className="flex items-start gap-3">
                <Shield className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-foreground">Security Notice</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    This system is for authorized law enforcement personnel only. 
                    All access attempts are logged and monitored.
                  </p>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="text-center text-xs text-muted-foreground">
              <p>© 2024 ForensicAI. All rights reserved.</p>
              <p className="mt-1">Version 2.1.0 - Classified System</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Login;