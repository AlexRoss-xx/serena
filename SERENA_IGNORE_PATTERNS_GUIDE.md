# Serena - Ignore Patterns Configuration Guide

## Overview

Configure which files/folders Serena should skip during indexing and symbol searches to improve performance on large projects.

---

## 📁 Configuration Location

**File:** `<project-root>/.serena/project.yml`

**Section:** `ignored_paths`

---

## 📝 Syntax (Same as .gitignore)

```yaml
ignored_paths:
  - "pattern1"
  - "pattern2"
  # Comments are allowed
```

### **Pattern Rules:**

- `*` = matches any characters except `/`
- `**` = matches any characters including `/` (recursive)
- `/` = directory separator (use forward slash even on Windows)
- `!` = negate pattern (include despite previous exclude)

---

## 🎯 Common Patterns for Pascal/Delphi Projects

### **Example 1: Basic Exclusions**

```yaml
ignored_paths:
  # Auto-generated files
  - "**/*_TLB.pas"          # Type library imports
  
  # Build artifacts
  - "**/__history/**"       # IDE history
  - "**/backup/**"          # Backups
  - "**/*.dcu"              # Compiled units
  - "**/*.ppu"              # FPC compiled units
  - "**/*.o"                # Object files
```

### **Example 2: Test Directories**

```yaml
ignored_paths:
  # Test files (search separately if needed)
  - "**/DUnit/**"
  - "**/Tests/**"
  - "**/Test/**"
  - "**/*Test.pas"          # Files ending with Test
  - "**/UD_*.pas"           # DUnit test files
```

### **Example 3: Third-Party Libraries**

```yaml
ignored_paths:
  # Large external libraries
  - "Common/Indy/**"
  - "Common/Jedi/**"
  - "Common/DevExpress*/**"  # DevExpress6, DevExpress7, etc.
  - "Common/TeeChart/**"
  - "Common/tp/**"           # TurboPower libraries
  - "**/External/**"
```

### **Example 4: Examples and Demos**

```yaml
ignored_paths:
  - "**/examples/**"
  - "**/Examples/**"
  - "**/demo/**"
  - "**/Demo/**"
  - "**/Sample/**"
  - "**/Samples/**"
```

### **Example 5: Specific Paths**

```yaml
ignored_paths:
  # Ignore specific subdirectories
  - "Profile/Accession.NET/**"       # .NET projects
  - "Common/DotNet/**"                # .NET assemblies
  - "STAGE_ROOT/**"                   # Staging area
  - "Tools/**"                        # Build tools
```

---

## ✅ Complete Example for Profile Project

```yaml
ignored_paths:
  # === Auto-generated files ===
  - "**/*_TLB.pas"              # Type libraries (111 files)
  
  # === Test directories ===
  - "**/DUnit/**"
  - "**/Tests/**"
  - "**/Test/**"
  
  # === Build artifacts ===
  - "**/__history/**"
  - "**/backup/**"
  - "**/__recovery/**"
  - "**/*.bak"
  - "**/*.dcu"
  - "**/*.o"
  - "**/*.ppu"
  
  # === Examples and demos ===
  - "**/examples/**"
  - "**/Examples/**"
  - "**/demo/**"
  - "**/Demo/**"
  - "**/Sample/**"
  
  # === Large third-party libraries ===
  # Uncomment as needed:
  # - "Common/Indy/**"
  # - "Common/Jedi/**"
  # - "Common/DevExpress*/**"
  # - "Common/TeeChart/**"
  # - "Common/tp/Abbrevia/**"
  # - "Common/tp/systools/**"
  # - "Common/tp/tpapro/**"
  
  # === .NET components ===
  # - "Profile/Accession.NET/**"
  # - "Common/DotNet/**"
  
  # === Staging/temporary ===
  # - "STAGE_ROOT/**"
  # - "Tools/**"
```

---

## 🔍 How to Find Slow Files to Ignore

### **Method 1: Check Serena Logs**

Look for files taking >50ms to reload:

```powershell
# Find slow files in latest log
Get-ChildItem "C:\Users\$env:USERNAME\.serena\logs" -Recurse -Filter "mcp_*.txt" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 | 
    ForEach-Object { Get-Content $_.FullName } | 
    Select-String -Pattern "Reloaded.*\d+ms" | 
    Where-Object { $_ -match "(\d+)ms" -and [int]$Matches[1] -gt 50 } | 
    Select-Object -First 20
```

### **Method 2: Check File Sizes**

Large files (>100 KB) are often slow to parse:

```powershell
# Find large Pascal files
Get-ChildItem "D:\Work\Intrahealth\repositories\profile" -Recurse -File -Filter "*.pas" | 
    Where-Object { $_.Length -gt 100KB } | 
    Sort-Object Length -Descending | 
    Select-Object -First 20 Name, @{N='SizeKB';E={[math]::Round($_.Length/1KB,1)}}
```

---

## ⚙️ How Ignore Patterns Work

### **What Gets Ignored:**

✅ **File/folder matching pattern:**
- Skipped during symbol indexing
- Not searched by `find_symbol` (unless path specified)
- Not shown in directory listings
- Reduces indexing time

### **What's NOT Affected:**

❌ **You can still access ignored files by:**
- Direct file path: `read_file(relative_path='DUnit/test.pas')`
- Explicit search: `search_for_pattern(relative_path='DUnit')`
- Directory listing with full path

---

## 📊 Performance Impact

For Profile project with recommended ignore patterns:

| Metric | Without Ignores | With Ignores | Improvement |
|--------|----------------|--------------|-------------|
| Files indexed | ~2000 | ~1500 | 25% reduction |
| Initial indexing | 10-15 min | 7-10 min | ~30% faster |
| Symbol searches | Timeout often | Complete in 1-5 min | Much more reliable |
| Type library files | Parsed (very slow) | Skipped | Major speed boost |

---

## 🔄 Applying Changes

### **After editing `.serena/project.yml`:**

**Option 1: Restart Language Server (Fast)**
```python
# In Serena chat (if restart_language_server is in included_optional_tools)
restart_language_server()
```

**Option 2: Re-activate Project (Recommended)**
```python
# In Serena chat
activate_project: D:\Work\Intrahealth\repositories\profile
```

**Option 3: Full Restart (Nuclear)**
```powershell
# Kill processes
taskkill /f /im pasls.exe

# Clear caches
Remove-Item "D:\...\profile\.serena\cache\pascal\*" -Force

# Restart Cursor and activate project
```

---

## 🧪 Testing Ignore Patterns

### **Verify Pattern is Working:**

```python
# Search in ignored directory - should return empty or fewer results
find_symbol(pattern='SomeSymbol', relative_path='Common/DUnit')

# Search in non-ignored directory - should work normally
find_symbol(pattern='SomeSymbol', relative_path='Common/Common/Kernel')
```

### **Count Indexed Files:**

Check Serena log after activation:

```
Found 214 Pascal source directories in profile
```

After adding ignores, this number should decrease.

---

## 💡 Tips

1. **Start Conservative** - Ignore only obvious candidates (tests, examples)
2. **Monitor Logs** - Find files taking >100ms to parse
3. **Add Incrementally** - Add slow patterns one at a time
4. **Keep .gitignore** - `ignore_all_files_in_gitignore: true` already excludes build dirs
5. **Document Why** - Add comments explaining each ignore pattern

---

## 🎯 Quick Setup for Profile Project

**Recommended minimal ignore list:**

```yaml
ignored_paths:
  - "**/*_TLB.pas"    # 111 files - auto-generated, never edited
  - "**/DUnit/**"     # Unit tests - search separately if needed
  - "**/examples/**"  # Examples - rarely edited
```

This alone will:
- ✅ Reduce indexed files by ~150
- ✅ Speed up full-project searches by 2-3x
- ✅ Prevent timeout on initialization

---

**Last Updated:** 2025-12-21  
**For:** Serena v0.1.4 with Pascal Language Server



