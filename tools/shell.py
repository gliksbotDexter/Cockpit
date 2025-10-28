from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import subprocess
import os
import sys
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from queue import Queue, Empty

@dataclass
class CommandResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    command: str
    truncated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_summary(self) -> str:
        status = "✅ SUCCESS" if self.success else f"❌ FAILED (exit code: {self.exit_code})"
        output = self.stdout or self.stderr
        preview = output[:500] + "..." if len(output) > 500 else output
        return f"""{status}
⏱️  Execution Time: {self.execution_time:.2f}s
💻 Command: {self.command}
📝 Output Preview:
{preview}"""

class AdvancedPowerShellTool:
    name = "powershell"
    description = """Advanced PowerShell execution with safety, monitoring, and enterprise features.
    Features:
    - Secure command validation
    - Real-time output streaming
    - Timeout management
    - Process monitoring
    - JSON output support
    - Session persistence
    - Error handling and recovery
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.security = cfg.get("security", {})
        self.history: List[CommandResult] = []
        self.max_history = cfg.get("max_history", 50)
        self.default_timeout = cfg.get("default_timeout", 300)  # 5 minutes
        self.ps_exe = self._find_powershell()
        
        # Security configurations
        self.allowlist = set(self.security.get("allowlist", []))
        self.denylist = set(self.security.get("denylist", [
            "remove-item", "del", "rm", "erase", "format",
            "stop-computer", "restart-computer", "shutdown",
            "invoke-webrequest", "wget", "curl", "bitsadmin",
            "certutil", "mshta", "regsvr32", "rundll32"
        ]))
        self.safe_mode = self.security.get("safe_mode", True)
        
        # Enterprise features
        self.session_id = f"session_{int(time.time())}"
        self.working_directory = Path(cfg.get("working_directory", ".")).resolve()
        
    def _find_powershell(self) -> str:
        """Find the best PowerShell executable"""
        candidates = [
            os.environ.get("PS_EXE"),
            "pwsh.exe",  # PowerShell Core
            "powershell.exe",  # Windows PowerShell
            "C:\\Program Files\\PowerShell\\7\\pwsh.exe",
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
        ]
        
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        
        return "powershell.exe"  # Fallback
    
    def _is_command_allowed(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if command is allowed based on security policies"""
        if not self.safe_mode:
            return True, None
            
        cmd_lower = command.lower().strip()
        
        # Check denylist
        for denied in self.denylist:
            if denied.lower() in cmd_lower:
                return False, f"Blocked by security policy: {denied}"
        
        # Check allowlist if configured
        if self.allowlist:
            for allowed in self.allowlist:
                if allowed.lower() in cmd_lower:
                    return True, None
            return False, "Command not in allowlist"
        
        return True, None
    
    def _sanitize_command(self, command: str) -> str:
        """Sanitize command for safe execution"""
        # Remove dangerous patterns
        dangerous_patterns = [
            ';', '&&', '||', '|', '`', '%',
            '$env:', 'iex', 'invoke-expression'
        ]
        
        sanitized = command
        for pattern in dangerous_patterns:
            if self.safe_mode and pattern in sanitized.lower():
                raise ValueError(f"Potentially dangerous pattern detected: {pattern}")
        
        return sanitized
    
    def _execute_with_streaming(self, command: str, timeout: int) -> CommandResult:
        """Execute command with real-time output streaming"""
        start_time = time.time()
        
        try:
            # Prepare PowerShell arguments
            ps_args = [
                self.ps_exe,
                "-NoLogo",
                "-NonInteractive", 
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command", command
            ]
            
            # Start process
            process = subprocess.Popen(
                ps_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.working_directory)
            )
            
            # Capture output with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(ps_args, timeout)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = CommandResult(
                success=process.returncode == 0,
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                exit_code=process.returncode,
                execution_time=execution_time,
                command=command
            )
            
            # Truncate very long outputs
            if len(result.stdout) > 10000 or len(result.stderr) > 10000:
                result.truncated = True
                if len(result.stdout) > 10000:
                    result.stdout = result.stdout[:10000] + "\n... [output truncated]"
                if len(result.stderr) > 10000:
                    result.stderr = result.stderr[:10000] + "\n... [error output truncated]"
            
            return result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return CommandResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                exit_code=-1,
                execution_time=execution_time,
                command=command
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return CommandResult(
                success=False,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                execution_time=execution_time,
                command=command
            )
    
    def run(self, command: str, timeout: Optional[int] = None, 
            format_output: str = "summary") -> str:
        """
        Execute PowerShell command with advanced features
        
        Args:
            command: PowerShell command to execute
            timeout: Execution timeout in seconds (defaults to config value)
            format_output: Output format ("summary", "detailed", "json", "raw")
        """
        # Validate inputs
        if not command or not command.strip():
            return "Error: Empty command provided"
        
        timeout = timeout or self.default_timeout
        
        # Security check
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return f"🚫 SECURITY BLOCKED: {reason}"
        
        # Sanitize command
        try:
            sanitized_cmd = self._sanitize_command(command)
        except ValueError as e:
            return f"🛡️  SANITIZATION ERROR: {str(e)}"
        
        # Execute command
        result = self._execute_with_streaming(sanitized_cmd, timeout)
        
        # Store in history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Format output
        if format_output == "json":
            return json.dumps(result.to_dict(), indent=2)
        elif format_output == "detailed":
            return f"""PowerShell Command Execution Report
=====================================
Session: {self.session_id}
Working Directory: {self.working_directory}
Command: {result.command}

📊 RESULT:
Status: {"✅ SUCCESS" if result.success else "❌ FAILED"}
Exit Code: {result.exit_code}
Execution Time: {result.execution_time:.2f} seconds
Truncated: {"Yes" if result.truncated else "No"}

📥 STDOUT:
{result.stdout or "(no standard output)"}

.Stderr:
{result.stderr or "(no error output)"}
"""
        elif format_output == "raw":
            return result.stdout or result.stderr or "(no output)"
        else:  # summary
            return result.to_summary()
    
    def get_system_info(self) -> str:
        """Get comprehensive system information"""
        info_commands = [
            "Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, TotalPhysicalMemory",
            "Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object LastBootUpTime, LocalDateTime",
            "Get-Process | Measure-Object | Select-Object Count",
            "Get-Service | Where-Object {$_.Status -eq 'Running'} | Measure-Object | Select-Object Count"
        ]
        
        results = []
        for cmd in info_commands:
            result = self._execute_with_streaming(cmd, 30)
            if result.success:
                results.append(result.stdout)
        
        return "\n--- System Information ---\n".join(results) if results else "Failed to retrieve system info"
    
    def get_process_list(self, filter_name: str = None) -> str:
        """Get running processes with optional filtering"""
        if filter_name:
            cmd = f"Get-Process | Where-Object {{$_.Name -like '*{filter_name}*'}} | Select-Object Name, Id, CPU, WorkingSet | Format-Table -AutoSize"
        else:
            cmd = "Get-Process | Select-Object Name, Id, CPU, WorkingSet | Sort-Object CPU -Descending | Select-Object -First 20 | Format-Table -AutoSize"
        
        result = self._execute_with_streaming(cmd, 60)
        return result.to_summary()
    
    def get_service_status(self, service_name: str = None) -> str:
        """Get service status"""
        if service_name:
            cmd = f"Get-Service -Name '{service_name}' | Select-Object Name, Status, DisplayName | Format-List"
        else:
            cmd = "Get-Service | Where-Object {$_.Status -eq 'Running'} | Select-Object Name, Status | Select-Object -First 10"
        
        result = self._execute_with_streaming(cmd, 30)
        return result.to_summary()
    
    def get_disk_usage(self) -> str:
        """Get disk usage information"""
        cmd = "Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free | Format-Table -AutoSize"
        result = self._execute_with_streaming(cmd, 30)
        return result.to_summary()
    
    def get_network_info(self) -> str:
        """Get network configuration information"""
        cmd = "Get-NetIPConfiguration | Select-Object InterfaceAlias, IPv4Address | Format-Table -AutoSize"
        result = self._execute_with_streaming(cmd, 30)
        return result.to_summary()
    
    def get_event_logs(self, log_name: str = "System", max_events: int = 10) -> str:
        """Get recent event logs"""
        cmd = f"Get-WinEvent -LogName {log_name} -MaxEvents {max_events} | Select-Object TimeCreated, LevelDisplayName, Id, Message | Format-Table -AutoSize"
        result = self._execute_with_streaming(cmd, 60)
        return result.to_summary()
    
    def run_advanced(self, action: str, **kwargs) -> str:
        """Run advanced system administration actions"""
        actions = {
            "system_info": self.get_system_info,
            "processes": lambda: self.get_process_list(kwargs.get("filter")),
            "services": lambda: self.get_service_status(kwargs.get("service_name")),
            "disk_usage": self.get_disk_usage,
            "network": self.get_network_info,
            "events": lambda: self.get_event_logs(
                kwargs.get("log_name", "System"),
                kwargs.get("max_events", 10)
            )
        }
        
        if action in actions:
            return actions[action]()
        else:
            return f"Unknown action: {action}. Available actions: {', '.join(actions.keys())}"
    
    def get_session_stats(self) -> str:
        """Get session statistics"""
        total_commands = len(self.history)
        successful = sum(1 for h in self.history if h.success)
        failed = total_commands - successful
        
        if self.history:
            avg_time = sum(h.execution_time for h in self.history) / len(self.history)
        else:
            avg_time = 0
        
        return f"""Session Statistics
================
Session ID: {self.session_id}
Total Commands: {total_commands}
Successful: {successful}
Failed: {failed}
Average Execution Time: {avg_time:.2f}s
Working Directory: {self.working_directory}"""

# Enhanced Shell Tool for non-PowerShell commands
class AdvancedShellTool:
    name = "shell"
    description = """Advanced shell command execution with Windows support.
    Features:
    - CMD and WSL integration
    - Batch script execution
    - Environment variable management
    - File operation safety
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.security = cfg.get("security", {})
        self.denylist = set(self.security.get("denylist", [
            "del", "erase", "format", "rd", "rmdir"
        ]))
        self.working_directory = Path(cfg.get("working_directory", ".")).resolve()
    
    def run(self, command: str, shell_type: str = "cmd", timeout: int = 60) -> str:
        """Execute shell command"""
        # Security check
        cmd_lower = command.lower()
        for denied in self.denylist:
            if denied.lower() in cmd_lower:
                return f"🚫 DENIED: {denied} blocked by security policy"
        
        try:
            if shell_type == "wsl":
                full_cmd = f"wsl {command}"
            else:  # cmd
                full_cmd = f"cmd /c {command}"
            
            start_time = time.time()
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_directory)
            )
            execution_time = time.time() - start_time
            
            output = result.stdout.strip() or result.stderr.strip()
            status = "✅ SUCCESS" if result.returncode == 0 else f"❌ FAILED (code: {result.returncode})"
            
            return f"""{status}
⏱️  Execution Time: {execution_time:.2f}s
💻 Command: {command}
📝 Output:
{output[:1000]}{'...' if len(output) > 1000 else ''}"""
            
        except subprocess.TimeoutExpired:
            return "⏰ TIMEOUT: Command took too long to execute"
        except Exception as e:
            return f"💥 ERROR: {str(e)}"

# Factory function
def create_windows_tools(cfg: Dict[str, Any]) -> List[Any]:
    """Create Windows-specific tools"""
    return [
        AdvancedPowerShellTool(cfg),
        AdvancedShellTool(cfg)
    ]

# Example configuration
WINDOWS_TOOL_CONFIG = {
    "security": {
        "denylist": [
            "remove-item", "del", "rm", "erase", "format",
            "stop-computer", "restart-computer", "shutdown",
             "wget", "bitsadmin"
        ],
        "allowlist": [],  # Empty means allow all (except denylist)
        
    },
    "default_timeout": 300,
    
    
}

