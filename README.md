# Pokemon Champions Assistant

一个配合雷电模拟器使用的本地宝可梦冠军对战助手。V1 只做截图识别、属性克制、粗略伤害估算和手动修正，不发送游戏点击指令，不读取或修改游戏数据。

## 功能

- 通过 ADB 发现 Android 模拟器并保存调试截图。
- 使用可配置 ROI 识别双方宝可梦名称，OCR 引擎可选。
- 支持 63 单打和 64 双打：双方 6 只队伍预览 + 当前场上 1v1 或 2v2 槽位。
- 根据本地 `data/` 数据展示属性弱点、抗性、免疫、换入候选和常见招式威胁。
- 提供简化伤害区间估算，适合快速判断局势。
- 提供 PySide6 桌面 UI；未安装 UI 依赖时，核心 CLI 和测试仍可运行。

## 安装

核心逻辑不强制安装第三方库：

```powershell
python -m champions_assistant --help
```

桌面界面和 OCR 建议安装：

```powershell
python -m pip install -e ".[ui,vision,test]"
```

如果暂时只想跑测试：

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

## CLI

```powershell
python -m champions_assistant run
python -m champions_assistant calibrate
python -m champions_assistant capture --out screenshots/sample.png
python -m champions_assistant analyze --self Pikachu --opponent Gyarados
python -m champions_assistant analyze --format doubles64 --self-team "Pikachu,Charizard,Gengar,Lucario,Dragonite,Sylveon" --opponent-team "Gyarados,Garchomp,Metagross,Venusaur,Incineroar,Flutter Mane" --self-active "Pikachu,Charizard" --opponent-active "Gyarados,Venusaur"
```

## 对手头像模板识别

右侧对手队伍在选人界面没有名字，所以需要用头像模板识别。建议把雷电模拟器固定为 `1920x1080`。

第一次采集模板：

```powershell
python -m champions_assistant capture --out screenshots/team_select.png
python -m champions_assistant harvest-templates --image screenshots/team_select.png --opponent "swampert,meganium,pelipper,basculegion,duraludon,garchomp"
```

离线测试识别效果：

```powershell
python -m champions_assistant recognize-preview --image screenshots/team_select.png
```

模板会保存在 `assets/pokemon_templates/<species_id>/preview_*.png`。同一只宝可梦可以保存多张模板，程序会取匹配分最高的一张。

## Windows 启动器

项目根目录提供 `PokemonChampionsAssistant.exe`，双击即可启动桌面 UI。它会自动寻找 Python，并检查 `PySide6` 和项目包是否可导入。

重新生成启动器：

```powershell
.\build_launcher.ps1
```

## 配置

默认配置在 `config/app.toml`：

- `adb_path`：ADB 可执行文件路径，默认 `adb`。
- `device_serial`：ADB 设备序列号，留空时自动选择唯一在线设备。
- `capture_interval_ms`：UI 自动刷新截图间隔。
- `default_format`：默认 `singles63`，也可设为 `doubles64`。
- `[roi.*]`：队伍预览、场上槽位、HP、回合区域的截图坐标。右侧对手 6 个预览槽位内置了 `1920x1080` 默认头像 ROI；旧的 `self_name/opponent_name` 会自动迁移到新的场上 1 号槽位。

## 数据

`data/` 下的数据是可替换的：

- `type_chart.json`：18 属性克制关系。
- `pokemon.json`：宝可梦基础信息和常见招式。
- `moves.json`：招式类型、分类和威力。
- `aliases.json`：中文、英文、小名到标准宝可梦 ID 的映射。

V1 附带的是基础样本数据，不宣称覆盖 Pokémon Champions 全量实装环境。
