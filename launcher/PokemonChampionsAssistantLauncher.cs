using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

internal static class PokemonChampionsAssistantLauncher
{
    private const string AppTitle = "Pokemon Champions Assistant";

    [STAThread]
    private static int Main()
    {
        Application.EnableVisualStyles();
        var appDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var python = FindPython(appDir);

        if (python == null)
        {
            MessageBox.Show(
                "没有找到可用的 Python 环境。\n\n请安装 Python 3.11+，并执行：\npython -m pip install -e \".[ui]\"\n\n也可以设置环境变量 CHAMPIONS_ASSISTANT_PYTHON 指向正确的 python.exe。",
                AppTitle,
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 1;
        }

        try
        {
            var process = new Process();
            process.StartInfo.FileName = python.WindowPath ?? python.ConsolePath;
            process.StartInfo.Arguments = "-m champions_assistant run";
            process.StartInfo.WorkingDirectory = appDir;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.CreateNoWindow = true;
            process.Start();
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show("启动失败：\n\n" + ex.Message, AppTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 3;
        }
    }

    private static ValidationResult ValidateEnvironment(string consolePython, string appDir)
    {
        try
        {
            var process = new Process();
            process.StartInfo.FileName = consolePython;
            process.StartInfo.Arguments = "-c \"import PySide6; import champions_assistant\"";
            process.StartInfo.WorkingDirectory = appDir;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.CreateNoWindow = true;
            process.StartInfo.RedirectStandardError = true;
            process.StartInfo.RedirectStandardOutput = true;
            process.Start();

            var exited = process.WaitForExit(8000);
            if (!exited)
            {
                try { process.Kill(); } catch { }
                return ValidationResult.Fail("Python 环境检查超时。");
            }

            if (process.ExitCode == 0)
            {
                return ValidationResult.Success();
            }

            var output = (process.StandardError.ReadToEnd() + "\n" + process.StandardOutput.ReadToEnd()).Trim();
            return ValidationResult.Fail(string.IsNullOrWhiteSpace(output) ? "Python 无法导入 UI 依赖。" : output);
        }
        catch (Exception ex)
        {
            return ValidationResult.Fail(ex.Message);
        }
    }

    private static PythonInstall FindPython(string appDir)
    {
        var candidates = new List<string>();
        var fromEnv = Environment.GetEnvironmentVariable("CHAMPIONS_ASSISTANT_PYTHON");
        AddIfFile(candidates, fromEnv);

        AddIfFile(candidates, Path.Combine(appDir, ".venv", "Scripts", "python.exe"));
        AddIfFile(candidates, Path.Combine(appDir, "venv", "Scripts", "python.exe"));

        AddRange(candidates, Where("python.exe"));
        AddRange(candidates, Where("py.exe"));

        foreach (var version in new[] { "314", "313", "312", "311" })
        {
            AddIfFile(candidates, @"C:\Python" + version + @"\python.exe");
            AddIfFile(candidates, Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs",
                "Python",
                "Python" + version,
                "python.exe"));
        }

        var failures = new List<string>();
        foreach (var candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var resolved = ResolvePython(candidate);
            if (resolved == null)
            {
                continue;
            }

            var validation = ValidateEnvironment(resolved.ConsolePath, appDir);
            if (validation.Ok)
            {
                return resolved;
            }

            failures.Add(resolved.ConsolePath + ": " + validation.Message);
        }

        if (failures.Count > 0)
        {
            MessageBox.Show(
                "找到了 Python，但没有一个能启动本软件：\n\n" + string.Join("\n\n", failures.Take(3).ToArray()) +
                "\n\n可在项目目录执行：\npython -m pip install -e \".[ui]\"",
                AppTitle,
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
        }

        return null;
    }

    private static PythonInstall ResolvePython(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        var fileName = Path.GetFileName(path);
        if (fileName.Equals("py.exe", StringComparison.OrdinalIgnoreCase))
        {
            return new PythonInstall(path, path);
        }

        if (!File.Exists(path))
        {
            return null;
        }

        var windowPath = Path.Combine(Path.GetDirectoryName(path) ?? "", "pythonw.exe");
        return new PythonInstall(path, File.Exists(windowPath) ? windowPath : path);
    }

    private static IEnumerable<string> Where(string executable)
    {
        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var part in path.Split(Path.PathSeparator))
        {
            if (string.IsNullOrWhiteSpace(part))
            {
                continue;
            }

            var candidate = Path.Combine(part.Trim('"'), executable);
            if (File.Exists(candidate))
            {
                yield return candidate;
            }
        }
    }

    private static void AddIfFile(List<string> values, string path)
    {
        if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
        {
            values.Add(path);
        }
    }

    private static void AddRange(List<string> values, IEnumerable<string> paths)
    {
        foreach (var path in paths)
        {
            AddIfFile(values, path);
        }
    }

    private sealed class PythonInstall
    {
        public PythonInstall(string consolePath, string windowPath)
        {
            ConsolePath = consolePath;
            WindowPath = windowPath;
        }

        public string ConsolePath { get; private set; }
        public string WindowPath { get; private set; }
    }

    private sealed class ValidationResult
    {
        private ValidationResult(bool ok, string message)
        {
            Ok = ok;
            Message = message;
        }

        public bool Ok { get; private set; }
        public string Message { get; private set; }

        public static ValidationResult Success()
        {
            return new ValidationResult(true, "");
        }

        public static ValidationResult Fail(string message)
        {
            return new ValidationResult(false, message);
        }
    }
}
