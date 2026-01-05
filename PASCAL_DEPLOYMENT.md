# Pascal Language Server Production Deployment Guide

## Overview

This document provides comprehensive guidance for deploying the Pascal Language Server (`pasls`) with Serena in production environments supporting large Delphi and Pascal codebases.

---

## Distribution Package

### Required Components

```
pasls/
├── pasls.exe           (1.8 MB)  - Pascal Language Server executable
└── sqlite3.dll         (2.5 MB)  - Required runtime dependency

fpc/3.3.1/
├── bin/i386-win32/     (91 MB)   - FPC compiler binaries
├── source/             (281 MB)  - FPC source code (required by CodeTools)
└── units/i386-win32/   (~50 MB)  - RTL/FCL packages only
```

**FPC Download:** https://drive.google.com/embeddedfolderview?id=0B3iIrj8df2EtOVBIQi1ZOGpSdVk&resourcekey=0-xApPRycoHouNaf4JUXUc8w#list

**Total Distribution Size:** Approximately 420 MB (trimmed) or 1.1 GB (full installation)

---

## Environment Configuration

### Required Environment Variables

All team members must configure the following user-level environment variables:

**Template (replace paths with your installation locations):**

```powershell
# 1. Pascal Language Server executable path (must include filename)
[Environment]::SetEnvironmentVariable("SERENA_PASLS_PATH", "<path-to-pasls>\pasls.exe", "User")

# 2. FPC compiler bin directory path (must point to bin folder)
[Environment]::SetEnvironmentVariable("SERENA_FPC_PATH", "<path-to-fpc>\bin\i386-win32", "User")

# 3. Verification
Write-Host "SERENA_PASLS_PATH: $([Environment]::GetEnvironmentVariable('SERENA_PASLS_PATH', 'User'))"
Write-Host "SERENA_FPC_PATH: $([Environment]::GetEnvironmentVariable('SERENA_FPC_PATH', 'User'))"
```

**Example (with specific paths):**

```powershell
# 1. Pascal Language Server executable path (must include filename)
[Environment]::SetEnvironmentVariable("SERENA_PASLS_PATH", "D:\Tools\pasls\pasls.exe", "User")

# 2. FPC compiler bin directory path (must point to bin folder)
[Environment]::SetEnvironmentVariable("SERENA_FPC_PATH", "C:\FPC\3.3.1\bin\i386-win32", "User")

# 3. Verification
Write-Host "SERENA_PASLS_PATH: $([Environment]::GetEnvironmentVariable('SERENA_PASLS_PATH', 'User'))"
Write-Host "SERENA_FPC_PATH: $([Environment]::GetEnvironmentVariable('SERENA_FPC_PATH', 'User'))"
```

**Important:** IDE restart is required after setting environment variables for changes to take effect.

---

## Symbol Database Configuration

### Overview

The symbol database provides significant performance improvements through SQLite-based persistent indexing:

- SQLite-based persistent indexing of all project symbols
- 10-100x performance improvement for symbol queries after initial indexing
- Automatic incremental updates on file modifications
- Team-wide sharing capability via `.serena/cache/pascal/symbols.db`

### Database Location

```
<project-root>/.serena/cache/pascal/symbols.db
```

### Initial Indexing Performance

On first project activation, `pasls` performs a complete project index:

- Small projects (~100 files): 30-60 seconds
- Medium projects (~500 files): 2-5 minutes
- Large projects (1000+ files): 5-15 minutes

**Monitoring Progress:** Consult Serena logs for "Reloaded <file> in Xms" messages to track indexing progress.

### Database Maintenance

- **Automatic Updates:** The database updates incrementally without manual intervention
- **Corruption Recovery:** Delete `symbols.db` and restart the language server to rebuild
- **Team Collaboration:** Commit `symbols.db` to version control for shared cache across team members

---

## Indexing Optimization

### Default Exclusions

The following directories are excluded by default in `pasls` via `is_ignored_dirname()`:

```yaml
- lib/
- backup/
- __history/
- dcu/
- bin/
- obj/
```

### Project-Specific Exclusions

Configure additional exclusions in `.serena/project.yml`:

```yaml
ignored_paths:
  # Auto-generated type libraries (large files, infrequently modified)
  - "**/*_TLB.pas"

  # Unit test directories (index separately if required)
  - "**/DUnit/**"
  - "**/Tests/**"

  # Project-specific exclusions
  - "**/ThirdParty/**"
  - "**/External/**"
```

---

## Best Practices for Large Projects

### 1. Implement Scoped Searches

**Inefficient Approach:**
```python
find_symbol(pattern='appointment', substring_matching=True)
# Scans entire project (1000+ files)
```

**Recommended Approach:**
```python
find_symbol(
    pattern='appointment',
    relative_path='Profile/Common/Kernel',
    substring_matching=True
)
# Scans targeted directory only (~50 files)
```

### 2. Utilize Symbol Database Capabilities

The following operations achieve near-instant performance after indexing:
- `find_symbol` with exact name matching
- `get_symbols_overview` for any indexed file
- `find_referencing_symbols` for dependency analysis

### 3. Monitor File Processing Performance

Review logs to identify files with processing time exceeding 100ms:

```
Reloaded D:\SomeFile.pas in 350ms
```

Consider adding slow-processing files to `ignored_paths` if they are infrequently modified.

---

## Troubleshooting

### Error: "TFPCUnitTo.GetConfigCache missing CompilerFilename"

**Root Cause:** `SERENA_FPC_PATH` environment variable is not configured or points to an incorrect location.

**Resolution:**
```powershell
# Verify FPC path points to bin directory (not FPC root)
[Environment]::GetEnvironmentVariable("SERENA_FPC_PATH", "User")
# Expected output: C:\FPC\3.3.1\bin\i386-win32
# Incorrect output: C:\FPC\3.3.1
```

### Error: "Request timed out (timeout=235)"

**Root Cause:** File complexity exceeds processing capacity or unscoped project-wide scan is being performed.

**Resolution:**
1. Implement scoped searches using `relative_path='Target/Directory'` parameter
2. Add slow-processing files to `ignored_paths` configuration
3. Use exact symbol name matching instead of substring search when possible

### Error: "include file not found 'UDefs.inc'"

**Root Cause:** Include paths are not properly configured in `castle-pasls.ini`.

**Resolution:** Verify that the INI file contains appropriate `-Fi` include path directives:
```ini
[extra_options]
options=-Fi<path-to-includes> -Fi<path-to-additional-includes>
```

**Example:**
```ini
[extra_options]
options=-FiD:\Common -FiD:\Common\Kernel
```

### Error: Symbol Database Corruption

**Symptoms:**
- Missing or incorrect symbol information
- Language server crashes during indexing operations

**Resolution:**
```powershell
# Remove corrupted database
Remove-Item "D:\profile\.serena\cache\pascal\symbols.db"
# Restart language server via Serena to trigger reindexing
```

---

## Performance Metrics

### Comparative Analysis (Without vs. With Symbol Database)

| Operation | Without Symbol DB | With Symbol DB |
|-----------|------------------|----------------|
| First activation | 3-5 minutes | 5-15 minutes (one-time indexing) |
| Subsequent activations | 10-30 seconds | <5 seconds |
| `find_symbol` (exact match) | 30-60 seconds | <1 second |
| `find_symbol` (substring, scoped) | 1-3 minutes | 5-10 seconds |
| `get_symbols_overview` | 1-2 seconds per file | <100ms per file |
| Workspace search (1000 files) | Timeout (>5 minutes) | 10-30 seconds (cached) |

---

## Deployment Checklist

- [ ] Install FPC 3.3.1 or compatible version
- [ ] Copy `pasls.exe` and `sqlite3.dll` to designated deployment location
- [ ] Configure `SERENA_PASLS_PATH` environment variable (include `.exe` extension)
- [ ] Configure `SERENA_FPC_PATH` environment variable (point to bin directory)
- [ ] Restart IDE/Cursor to apply environment changes
- [ ] Activate project in Serena
- [ ] Allow initial symbol database indexing to complete
- [ ] Configure `ignored_paths` for large or slow-processing files
- [ ] Commit `symbols.db` to version control (optional, recommended for team sharing)
- [ ] Distribute environment configuration to team members

---

## Updates and Maintenance

### Updating Pascal Language Server

1. Replace `pasls.exe` in the deployment directory with the new version
2. Maintain existing `SERENA_PASLS_PATH` environment variable (no changes required)
3. Restart Cursor/IDE
4. Symbol database will automatically migrate to new version if necessary

### Updating Serena

```powershell
cd D:\Work\serena
uv sync
# Restart Cursor MCP server
```

### Cache Reset Procedure

```powershell
# Complete cache reset (use when troubleshooting persistent issues)
Remove-Item "D:\profile\.serena\cache\pascal\*" -Recurse
# Restart language server to rebuild cache
```

---

## Support Resources

### Serena Integration Issues

- **Log Files:** `C:\Users\<username>\.serena\logs\`
- **Environment Verification:** Confirm all required environment variables are properly configured
- **Minimal Testing:** Validate configuration with a minimal test project before deploying to production

---

## Deployment Validation Criteria

Successful deployment is confirmed when the following criteria are met:

1. Project activation completes in under 10 seconds (after initial indexing)
2. `find_symbol` operations with exact name matching complete in under 2 seconds
3. No "missing CompilerFilename" errors appear in log files
4. No timeout errors occur for scoped symbol searches
5. `symbols.db` file exists and increases in size during indexing
6. Subsequent sessions utilize cached symbols without reindexing

---

## Document Information

**Version:** 1.0
**Last Updated:** 2025-12-21
**Tested Configuration:** FPC 3.3.1, Serena 0.1.4, pasls (genericptr fork)




