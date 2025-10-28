from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os
import stat
import hashlib
import json
import mimetypes
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class FileInfo:
    name: str
    size: int
    modified: str
    created: str
    is_directory: bool
    permissions: str
    mime_type: str
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AdvancedFileReadTool:
    name = "file_operations"
    description = """Advanced file operations with read, write, search, and analysis capabilities.
    Features:
    - Safe file reading with multiple encodings
    - File metadata and analysis
    - Directory listing and search
    - Content preview and filtering
    - Hash generation
    - Binary file handling
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.security = cfg.get("security", {})
        self.max_file_size = cfg.get("max_file_size", 100 * 1024 * 1024)  # 100MB default
        self.working_directory = Path(cfg.get("working_directory", ".")).resolve()
        self.deny_patterns = set(self.security.get("deny_patterns", [
            "*.exe", "*.dll", "*.bat", "*.ps1", "*.vbs"
        ]))
        
    def _is_path_safe(self, path_str: str) -> bool:
        """Check if path is safe to access"""
        path = Path(path_str).resolve()
        
        # Check if within working directory (if restricted)
        if self.security.get("restrict_to_working_dir", False):
            try:
                path.relative_to(self.working_directory)
            except ValueError:
                return False
        
        # Check deny patterns
        for pattern in self.deny_patterns:
            if path.match(pattern):
                return False
                
        return True
    
    def _get_file_info(self, path: Path) -> FileInfo:
        """Get comprehensive file information"""
        stat_info = path.stat()
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        
        # Get permissions
        mode = stat_info.st_mode
        perms = ""
        perms += "r" if mode & stat.S_IRUSR else "-"
        perms += "w" if mode & stat.S_IWUSR else "-"
        perms += "x" if mode & stat.S_IXUSR else "-"
        perms += "r" if mode & stat.S_IRGRP else "-"
        perms += "w" if mode & stat.S_IWGRP else "-"
        perms += "x" if mode & stat.S_IXGRP else "-"
        perms += "r" if mode & stat.S_IROTH else "-"
        perms += "w" if mode & stat.S_IWOTH else "-"
        perms += "x" if mode & stat.S_IXOTH else "-"
        
        return FileInfo(
            name=path.name,
            size=stat_info.st_size,
            modified=datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            created=datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            is_directory=path.is_dir(),
            permissions=perms,
            mime_type=mime_type or "unknown"
        )
    
    def read_file(self, path: str, encoding: str = "auto", 
                  preview_lines: int = None, start_line: int = 0) -> str:
        """Read file with advanced options"""
        try:
            file_path = Path(path).resolve()
            
            # Security check
            if not self._is_path_safe(str(file_path)):
                return f"🚫 ACCESS DENIED: {path}"
            
            if not file_path.exists():
                return f"❌ FILE NOT FOUND: {path}"
            
            if file_path.is_dir():
                return f"📁 {path} is a directory. Use 'list_directory' instead."
            
            # Check file size
            size = file_path.stat().st_size
            if size > self.max_file_size:
                return f"📏 FILE TOO LARGE: {size} bytes (max: {self.max_file_size})"
            
            # Get file info
            info = self._get_file_info(file_path)
            
            # Read content
            if encoding == "auto":
                encodings = ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1']
                content = None
                used_encoding = None
                
                for enc in encodings:
                    try:
                        content = file_path.read_text(encoding=enc)
                        used_encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    return "🔤 UNSUPPORTED ENCODING: Could not decode file"
            else:
                content = file_path.read_text(encoding=encoding)
                used_encoding = encoding
            
            # Handle preview
            if preview_lines:
                lines = content.splitlines()
                total_lines = len(lines)
                end_line = min(start_line + preview_lines, total_lines)
                preview_content = '\n'.join(lines[start_line:end_line])
                more_info = f"\n📄 Lines {start_line+1}-{end_line} of {total_lines}"
            else:
                preview_content = content
                more_info = ""
                if len(content) > 5000:
                    preview_content = content[:5000] + "\n... [content truncated]"
            
            return f"""📄 File Read: {path}
💾 Size: {info.size} bytes
🔤 Encoding: {used_encoding}
🕐 Modified: {info.modified}
🔐 Permissions: {info.permissions}
 MIME: {info.mime_type}{more_info}

📝 Content:
{preview_content}"""
            
        except PermissionError:
            return f"🔒 PERMISSION DENIED: Cannot read {path}"
        except Exception as e:
            return f"💥 READ ERROR: {str(e)}"
    
    def write_file(self, path: str, content: str, encoding: str = "utf-8",
                   backup: bool = False, append: bool = False) -> str:
        """Write file with advanced options"""
        try:
            file_path = Path(path).resolve()
            
            # Security check
            if not self._is_path_safe(str(file_path)):
                return f"🚫 ACCESS DENIED: {path}"
            
            # Create directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Backup existing file
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + f".backup.{int(datetime.now().timestamp())}")
                shutil.copy2(file_path, backup_path)
                backup_msg = f" (backed up to {backup_path.name})"
            else:
                backup_msg = ""
            
            # Write content
            mode = "a" if append else "w"
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            action = "Appended to" if append else "Wrote"
            return f"✅ {action} {file_path}{backup_msg} ({len(content)} bytes)"
            
        except PermissionError:
            return f"🔒 PERMISSION DENIED: Cannot write to {path}"
        except Exception as e:
            return f"💥 WRITE ERROR: {str(e)}"
    
    def list_directory(self, path: str = ".", recursive: bool = False,
                      pattern: str = None, show_hidden: bool = False) -> str:
        """List directory contents with filtering"""
        try:
            dir_path = Path(path).resolve()
            
            if not dir_path.exists():
                return f"❌ DIRECTORY NOT FOUND: {path}"
            
            if not dir_path.is_dir():
                return f"📄 {path} is a file. Use 'read_file' instead."
            
            # Get items
            if recursive:
                items = list(dir_path.rglob("*")) if show_hidden else [
                    item for item in dir_path.rglob("*") 
                    if not item.name.startswith('.')
                ]
            else:
                items = list(dir_path.iterdir()) if show_hidden else [
                    item for item in dir_path.iterdir() 
                    if not item.name.startswith('.')
                ]
            
            # Filter by pattern
            if pattern:
                items = [item for item in items if item.match(pattern)]
            
            # Sort items
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            # Get info for each item
            file_infos = []
            for item in items[:100]:  # Limit to 100 items
                try:
                    info = self._get_file_info(item)
                    file_infos.append(info)
                except Exception:
                    continue
            
            # Format output
            if not file_infos:
                return f"📂 {path} is empty"
            
            lines = [f"📂 Directory Listing: {path}"]
            if recursive:
                lines.append("(recursive)")
            if pattern:
                lines.append(f"(pattern: {pattern})")
            
            lines.append("")
            lines.append("TYPE NAME                 SIZE       MODIFIED")
            lines.append("-" * 50)
            
            for info in file_infos:
                type_icon = "📁" if info.is_directory else "📄"
                size_str = "<DIR>" if info.is_directory else f"{info.size:,}"
                lines.append(f"{type_icon} {info.name:<20} {size_str:<10} {info.modified.split('T')[0]}")
            
            if len(items) > 100:
                lines.append(f"\n... and {len(items) - 100} more items")
            
            return "\n".join(lines)
            
        except PermissionError:
            return f"🔒 PERMISSION DENIED: Cannot list {path}"
        except Exception as e:
            return f"💥 LIST ERROR: {str(e)}"
    
    def search_files(self, directory: str, search_term: str,
                    file_pattern: str = "*", case_sensitive: bool = False) -> str:
        """Search for files containing specific text."""
        try:
            dir_path = Path(directory).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                return f"? INVALID DIRECTORY: {directory}"

            matches: List[tuple[Path, Optional[int], str]] = []
            max_size = 10 * 1024 * 1024  # 10MB limit
            normalized_term = search_term if case_sensitive else search_term.lower()

            for file_path in dir_path.rglob(file_pattern):
                if not file_path.is_file():
                    continue
                try:
                    if file_path.stat().st_size > max_size:
                        continue
                except OSError:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                haystack = content if case_sensitive else content.lower()
                if normalized_term not in haystack:
                    continue

                snippet = ""
                line_no: Optional[int] = None
                for idx_line, line in enumerate(content.splitlines(), start=1):
                    candidate = line if case_sensitive else line.lower()
                    if normalized_term in candidate:
                        line_no = idx_line
                        snippet = line.strip()
                        break

                matches.append((file_path, line_no, snippet))
                if len(matches) >= 20:
                    break

            if not matches:
                return f"? No matches for '{search_term}' in {directory}"

            lines = [f"?? Matches for '{search_term}' in {directory}", '-' * 60]
            for path_item, line_no, snippet in matches:
                location = f"{path_item}:{line_no}" if line_no else str(path_item)
                preview = snippet[:200]
                lines.append(f"- {location} :: {preview}")

            if len(matches) >= 20:
                lines.append("... results truncated at 20 items")

            return '\n'.join(lines)

        except PermissionError:
            return f"?? PERMISSION DENIED: Cannot search {directory}"
        except Exception as e:
            return f"?? SEARCH ERROR: {str(e)}"

