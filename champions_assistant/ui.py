from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .adb import AdbClient, AdbError
from .config import AppConfig, ROI_KEYS, save_config
from .damage import DamageCalculator
from .data_loader import DataRepository
from .health import build_health_report
from .models import (
    BattleFormat,
    BattleSnapshot,
    FieldSlot,
    PokemonIdentity,
    Rect,
    TeamSlot,
    update_field_slot,
    update_team_slot,
)
from .ocr import BattleRecognizer
from .preview_recognition import accepted_count, recognize_opponent_preview
from .recommender import build_recommendations
from .roi import VisionDependencyError

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    raise


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.repository = DataRepository(config.data_dir)
        self.adb = AdbClient(config.adb_path, config.device_serial)
        self.recognizer = BattleRecognizer(self.repository, config)
        self.snapshot = BattleSnapshot.empty(config.default_format)
        self._combo_lookup: dict[QtWidgets.QComboBox, tuple[str, str, int]] = {}
        self._syncing = False
        self.setWindowTitle("Pokemon Champions Assistant")
        self.resize(1180, 760)
        if config.ui.always_on_top:
            self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self._build_ui()
        self._populate_data()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.refresh_capture)
        self._timer.start(max(500, config.capture_interval_ms))
        QtCore.QTimer.singleShot(0, self.show_health_summary)

    def _build_ui(self) -> None:
        toolbar = QtWidgets.QToolBar("Main")
        toolbar.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(toolbar)
        refresh_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
        self.refresh_action = toolbar.addAction(refresh_icon, "刷新")
        self.refresh_action.triggered.connect(self.refresh_capture)
        open_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
        self.test_image_action = toolbar.addAction(open_icon, "测试图片")
        self.test_image_action.triggered.connect(self.test_local_image)
        health_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation)
        self.health_action = toolbar.addAction(health_icon, "环境检查")
        self.health_action.triggered.connect(self.show_health_summary)
        self.always_top_action = toolbar.addAction("置顶")
        self.always_top_action.setCheckable(True)
        self.always_top_action.setChecked(self.config.ui.always_on_top)
        self.always_top_action.triggered.connect(self._toggle_always_on_top)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.status_label = QtWidgets.QLabel("就绪。不会向游戏发送点击或输入指令。")
        layout.addWidget(self.status_label)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self._build_current_tab()
        self._build_recommendations_tab()
        self._build_damage_tab()
        self._build_team_tab()

    def _build_current_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("模式"))
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItem("63 单打", BattleFormat.SINGLES_63.value)
        self.format_combo.addItem("64 双打", BattleFormat.DOUBLES_64.value)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        mode_row.addWidget(self.format_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        teams_layout = QtWidgets.QHBoxLayout()
        self.player_team_combos = self._make_team_group("己方队伍", "player")
        self.opponent_team_combos = self._make_team_group("对方队伍", "opponent")
        teams_layout.addWidget(self.player_team_combos["group"])
        teams_layout.addWidget(self.opponent_team_combos["group"])
        layout.addLayout(teams_layout, 2)

        active_layout = QtWidgets.QHBoxLayout()
        self.player_active_combos = self._make_active_group("己方场上", "player")
        self.opponent_active_combos = self._make_active_group("对方场上", "opponent")
        active_layout.addWidget(self.player_active_combos["group"])
        active_layout.addWidget(self.opponent_active_combos["group"])
        layout.addLayout(active_layout, 1)

        buttons = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("应用手动修正")
        apply_button.clicked.connect(self.apply_manual_selection)
        buttons.addWidget(apply_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.meta_label = QtWidgets.QLabel("OCR 未运行。首次使用建议先执行 calibrate 配置 ROI。")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)
        self.tabs.addTab(tab, "当前识别")

    def _make_team_group(self, title: str, side: str) -> dict[str, object]:
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)
        combos: list[QtWidgets.QComboBox] = []
        for index in range(1, 7):
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.currentIndexChanged.connect(self.apply_manual_selection)
            self._combo_lookup[combo] = ("team", side, index)
            combos.append(combo)
            grid.addWidget(QtWidgets.QLabel(str(index)), index - 1, 0)
            grid.addWidget(combo, index - 1, 1)
        grid.setColumnStretch(1, 1)
        return {"group": group, "combos": combos}

    def _make_active_group(self, title: str, side: str) -> dict[str, object]:
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)
        combos: list[QtWidgets.QComboBox] = []
        type_labels: list[QtWidgets.QLabel] = []
        hp_labels: list[QtWidgets.QLabel] = []
        for index in range(1, 3):
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.currentIndexChanged.connect(self.apply_manual_selection)
            self._combo_lookup[combo] = ("active", side, index)
            type_label = QtWidgets.QLabel("-")
            hp_label = QtWidgets.QLabel("")
            combos.append(combo)
            type_labels.append(type_label)
            hp_labels.append(hp_label)
            grid.addWidget(QtWidgets.QLabel(str(index)), index - 1, 0)
            grid.addWidget(combo, index - 1, 1)
            grid.addWidget(type_label, index - 1, 2)
            grid.addWidget(hp_label, index - 1, 3)
        grid.setColumnStretch(1, 1)
        return {"group": group, "combos": combos, "type_labels": type_labels, "hp_labels": hp_labels}

    def _build_recommendations_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.recommendations_text = QtWidgets.QTextEdit()
        self.recommendations_text.setReadOnly(True)
        layout.addWidget(self.recommendations_text)
        self.tabs.addTab(tab, "对战提示")

    def _build_damage_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(tab)
        self.damage_attacker = QtWidgets.QComboBox()
        self.damage_defender = QtWidgets.QComboBox()
        self.damage_move = QtWidgets.QComboBox()
        self.damage_move.setEditable(True)
        layout.addWidget(QtWidgets.QLabel("攻击方"), 0, 0)
        layout.addWidget(self.damage_attacker, 0, 1)
        layout.addWidget(QtWidgets.QLabel("防守方"), 1, 0)
        layout.addWidget(self.damage_defender, 1, 1)
        layout.addWidget(QtWidgets.QLabel("招式"), 2, 0)
        layout.addWidget(self.damage_move, 2, 1)
        calculate = QtWidgets.QPushButton("计算")
        calculate.clicked.connect(self.calculate_damage)
        layout.addWidget(calculate, 3, 1)
        self.damage_result = QtWidgets.QLabel("-")
        self.damage_result.setWordWrap(True)
        layout.addWidget(self.damage_result, 4, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        self.tabs.addTab(tab, "伤害计算")

    def _build_team_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.team_table = QtWidgets.QTableWidget(0, 4)
        self.team_table.setHorizontalHeaderLabels(["宝可梦", "属性", "常见招式", "速度"])
        self.team_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.team_table)
        self.tabs.addTab(tab, "队伍数据")

    def _populate_data(self) -> None:
        self.pokemon_items = self.repository.all_pokemon()
        all_combos = self._all_slot_combos()
        for combo in all_combos:
            self._fill_pokemon_combo(combo, include_unknown=True)

        self.damage_move.clear()
        for move in self.repository.moves_by_name.values():
            label = f"{move.name_zh} / {move.name}" if move.name_zh else move.name
            self.damage_move.addItem(label, move.name)

        self.team_table.setRowCount(len(self.pokemon_items))
        for row, identity in enumerate(self.pokemon_items):
            species_id = identity.species_id or ""
            stats = self.repository.base_stats(species_id)
            moves = ", ".join(move.name for move in self.repository.moves_for_pokemon(identity))
            type_labels = "/".join(self.repository.type_chart.label(t, self.config.language) for t in identity.types)
            self.team_table.setItem(row, 0, QtWidgets.QTableWidgetItem(self.repository.pokemon_label(species_id, self.config.language)))
            self.team_table.setItem(row, 1, QtWidgets.QTableWidgetItem(type_labels))
            self.team_table.setItem(row, 2, QtWidgets.QTableWidgetItem(moves))
            self.team_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(stats.get("speed", "-"))))
        self.team_table.resizeColumnsToContents()
        self._sync_ui_from_snapshot()
        self.apply_manual_selection()

    def _fill_pokemon_combo(self, combo: QtWidgets.QComboBox, *, include_unknown: bool) -> None:
        combo.blockSignals(True)
        combo.clear()
        if include_unknown:
            combo.addItem("未识别", "")
        for identity in self.pokemon_items:
            combo.addItem(self.repository.pokemon_label(identity.species_id or "", self.config.language), identity.species_id)
        combo.blockSignals(False)

    def refresh_capture(self) -> None:
        try:
            image_bytes = self.adb.capture_screenshot()
            self.snapshot = self.recognizer.recognize(
                image_bytes,
                previous=self.snapshot,
                battle_format=self.snapshot.battle_format,
            )
            self._sync_ui_from_snapshot()
            self.status_label.setText("截图已刷新。")
            known_player = sum(1 for slot in self.snapshot.player_active if slot.pokemon.is_known)
            known_opponent = sum(1 for slot in self.snapshot.opponent_active if slot.pokemon.is_known)
            opponent_templates = sum(1 for slot in self.snapshot.opponent_team if slot.pokemon.source == "template")
            self.meta_label.setText(
                f"OCR: {self.recognizer.engine.engine_name}; "
                f"场上识别：己方 {known_player}/{self.snapshot.active_slots_per_side}，"
                f"对方 {known_opponent}/{self.snapshot.active_slots_per_side}；"
                f"对手预览模板识别 {opponent_templates}/6"
            )
            self.apply_manual_selection()
        except (AdbError, RuntimeError, ValueError) as exc:
            self.status_label.setText(f"截图刷新失败：{exc}")

    def test_local_image(self) -> None:
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择游戏截图",
            str(self.config.screenshots_dir),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if not image_path:
            return
        path = Path(image_path)
        try:
            image_bytes = path.read_bytes()
            results = recognize_opponent_preview(self.config, self.repository, image_bytes)
        except (OSError, VisionDependencyError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "图片识别失败", f"{path.name}\n\n{exc}")
            return

        self._show_preview_recognition_results(path, image_bytes, results)
        self.status_label.setText(f"本地图片测试完成：{path.name}")

    def show_health_summary(self) -> None:
        report = build_health_report(self.config)
        text = "\n".join(report.lines())
        icon = QtWidgets.QMessageBox.Icon.Information if report.blocking_ok else QtWidgets.QMessageBox.Icon.Warning
        QtWidgets.QMessageBox(
            icon,
            "环境检查",
            text,
            QtWidgets.QMessageBox.StandardButton.Ok,
            self,
        ).exec()
        if report.blocking_ok:
            self.status_label.setText(f"环境检查完成：{report.warnings} 个提醒。")
        else:
            self.status_label.setText("环境检查发现缺项，请查看提示。")

    def _show_preview_recognition_results(self, image_path: Path, image_bytes: bytes, results) -> None:
        accepted = accepted_count(results)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("本地图片识别测试")
        dialog.resize(1080, 760)
        layout = QtWidgets.QVBoxLayout(dialog)

        summary = QtWidgets.QLabel(f"{image_path.name}\n识别率：{accepted}/{len(results)}")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

        original_label = QtWidgets.QLabel()
        original_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        original_label.setMinimumHeight(220)
        original_label.setPixmap(self._scaled_pixmap_from_bytes(image_bytes, QtCore.QSize(1000, 300)))
        splitter.addWidget(original_label)

        crop_panel = QtWidgets.QWidget()
        crop_layout = QtWidgets.QGridLayout(crop_panel)
        crop_layout.setContentsMargins(0, 0, 0, 0)
        for index, result in enumerate(results):
            item = QtWidgets.QWidget()
            item_layout = QtWidgets.QVBoxLayout(item)
            item_layout.setContentsMargins(4, 4, 4, 4)
            crop_label = QtWidgets.QLabel()
            crop_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            crop_label.setFixedSize(150, 120)
            crop_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            crop_label.setPixmap(self._scaled_pixmap_from_bytes(result.crop_bytes, QtCore.QSize(140, 105)))
            caption = QtWidgets.QLabel(
                f"{result.slot_index}: {result.label}\n"
                f"{result.confidence:.3f} {result.status}\n"
                f"ROI {result.crop_rect.x},{result.crop_rect.y},{result.crop_rect.width},{result.crop_rect.height}"
            )
            caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            caption.setWordWrap(True)
            item_layout.addWidget(crop_label)
            item_layout.addWidget(caption)
            crop_layout.addWidget(item, 0, index)
        crop_scroll = QtWidgets.QScrollArea()
        crop_scroll.setWidgetResizable(True)
        crop_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        crop_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        crop_scroll.setWidget(crop_panel)
        splitter.addWidget(crop_scroll)

        table = QtWidgets.QTableWidget(len(results), 5)
        table.setHorizontalHeaderLabels(["槽位", "识别结果", "置信度", "状态", "模板"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        for row, result in enumerate(results):
            template = str(result.template_path) if result.template_path else "-"
            values = [
                str(result.slot_index),
                result.label,
                f"{result.confidence:.3f}",
                result.status,
                template,
            ]
            for column, value in enumerate(values):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        table.horizontalHeader().setStretchLastSection(True)
        table.resizeColumnsToContents()
        splitter.addWidget(table)
        splitter.setSizes([260, 180, 260])

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _scaled_pixmap_from_bytes(self, image_bytes: bytes, size: QtCore.QSize) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(image_bytes)
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    def apply_manual_selection(self) -> None:
        if self._syncing:
            return
        fmt = BattleFormat.parse(self.format_combo.currentData() or BattleFormat.SINGLES_63.value)
        snapshot = self.snapshot.with_format(fmt)

        for index, combo in enumerate(self.player_team_combos["combos"], start=1):
            pokemon = self._identity_from_combo(combo)
            snapshot = replace(
                snapshot,
                player_team=update_team_slot(snapshot.player_team, index, pokemon, selected=index <= fmt.selected_team_size),
            )
        for index, combo in enumerate(self.opponent_team_combos["combos"], start=1):
            pokemon = self._identity_from_combo(combo)
            snapshot = replace(
                snapshot,
                opponent_team=update_team_slot(snapshot.opponent_team, index, pokemon, selected=index <= fmt.selected_team_size),
            )
        for index, combo in enumerate(self.player_active_combos["combos"], start=1):
            pokemon = self._identity_from_combo(combo)
            snapshot = replace(snapshot, player_active=update_field_slot(snapshot.player_active, index, pokemon))
        for index, combo in enumerate(self.opponent_active_combos["combos"], start=1):
            pokemon = self._identity_from_combo(combo)
            snapshot = replace(snapshot, opponent_active=update_field_slot(snapshot.opponent_active, index, pokemon))

        self.snapshot = snapshot
        self._update_active_visibility()
        self._update_type_labels()
        self._update_recommendations()
        self._refresh_damage_participants()

    def calculate_damage(self) -> None:
        attacker = self._identity_from_combo(self.damage_attacker)
        defender = self._identity_from_combo(self.damage_defender)
        move_name = self.damage_move.currentData() or self.damage_move.currentText().split("/")[-1].strip()
        move = self.repository.moves_by_name.get(move_name)
        if not move:
            self.damage_result.setText("未知招式。")
            return
        estimate = DamageCalculator(self.repository).estimate(attacker, defender, move)
        notes = ", ".join(estimate.notes) if estimate.notes else "no extra modifiers"
        self.damage_result.setText(
            f"{estimate.move_name}: {estimate.damage_min}-{estimate.damage_max} "
            f"({estimate.percent_min:.1f}%-{estimate.percent_max:.1f}%), "
            f"属性倍率 x{estimate.type_multiplier:g}, {notes}"
        )

    def _identity_from_combo(self, combo: QtWidgets.QComboBox) -> PokemonIdentity:
        species_id = combo.currentData()
        if species_id:
            return self.repository.identity_for_id(str(species_id), source="manual")
        text = combo.currentText().strip()
        if not text or text == "未识别":
            return PokemonIdentity(source="manual")
        return self.repository.resolve_pokemon(text, source="manual")

    def _sync_ui_from_snapshot(self) -> None:
        self._syncing = True
        try:
            index = self.format_combo.findData(self.snapshot.battle_format.value)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)
            self._sync_combo_group(self.player_team_combos["combos"], self.snapshot.player_team)
            self._sync_combo_group(self.opponent_team_combos["combos"], self.snapshot.opponent_team)
            self._sync_combo_group(self.player_active_combos["combos"], self.snapshot.player_active)
            self._sync_combo_group(self.opponent_active_combos["combos"], self.snapshot.opponent_active)
            self._update_active_visibility()
            self._update_type_labels()
            self._refresh_damage_participants()
        finally:
            self._syncing = False

    def _sync_combo_group(self, combos: list[QtWidgets.QComboBox], slots) -> None:
        for combo, slot in zip(combos, slots):
            self._sync_combo(combo, slot.pokemon)

    def _sync_combo(self, combo: QtWidgets.QComboBox, identity: PokemonIdentity) -> None:
        combo.blockSignals(True)
        try:
            data = identity.species_id or ""
            index = combo.findData(data)
            if index >= 0:
                combo.setCurrentIndex(index)
            elif identity.name and identity.name != "Unknown":
                combo.setEditText(identity.name)
        finally:
            combo.blockSignals(False)

    def _update_active_visibility(self) -> None:
        active_count = self.snapshot.active_slots_per_side
        for group in (self.player_active_combos, self.opponent_active_combos):
            combos = group["combos"]
            labels = group["type_labels"]
            hp_labels = group["hp_labels"]
            for index, combo in enumerate(combos, start=1):
                visible = index <= active_count
                combo.setVisible(visible)
                labels[index - 1].setVisible(visible)
                hp_labels[index - 1].setVisible(visible)

    def _update_type_labels(self) -> None:
        for group, slots in (
            (self.player_active_combos, self.snapshot.player_active),
            (self.opponent_active_combos, self.snapshot.opponent_active),
        ):
            type_labels = group["type_labels"]
            hp_labels = group["hp_labels"]
            for index, slot in enumerate(slots):
                type_labels[index].setText(self._types_text(slot.pokemon))
                hp_labels[index].setText(slot.hp_text)

    def _types_text(self, identity: PokemonIdentity) -> str:
        if not identity.types:
            return "-"
        return "/".join(self.repository.type_chart.label(t, self.config.language) for t in identity.types)

    def _update_recommendations(self) -> None:
        recommendations = build_recommendations(self.snapshot, self.repository, self.config.language)
        chunks = []
        for item in recommendations:
            chunks.append(f"[{item.title}]\n{item.reason}\n{item.action}\n")
        self.recommendations_text.setPlainText("\n".join(chunks).strip())

    def _refresh_damage_participants(self) -> None:
        current_attacker = self.damage_attacker.currentData()
        current_defender = self.damage_defender.currentData()
        self.damage_attacker.blockSignals(True)
        self.damage_defender.blockSignals(True)
        self.damage_attacker.clear()
        self.damage_defender.clear()

        player_slots = list(self.snapshot.player_active) + list(self.snapshot.player_team)
        opponent_slots = list(self.snapshot.opponent_active) + list(self.snapshot.opponent_team)
        self._add_damage_slots(self.damage_attacker, player_slots)
        self._add_damage_slots(self.damage_defender, opponent_slots)
        if current_attacker:
            self._set_combo_data(self.damage_attacker, current_attacker)
        if current_defender:
            self._set_combo_data(self.damage_defender, current_defender)
        self.damage_attacker.blockSignals(False)
        self.damage_defender.blockSignals(False)

    def _add_damage_slots(self, combo: QtWidgets.QComboBox, slots: list[FieldSlot | TeamSlot]) -> None:
        seen: set[str] = set()
        for slot in slots:
            pokemon = slot.pokemon
            if not pokemon.is_known or not pokemon.species_id or pokemon.species_id in seen:
                continue
            seen.add(pokemon.species_id)
            combo.addItem(f"{slot.label} {self.repository.pokemon_label(pokemon.species_id, self.config.language)}", pokemon.species_id)
        if combo.count() == 0:
            combo.addItem("未识别", "")

    def _set_combo_data(self, combo: QtWidgets.QComboBox, data: object) -> None:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _all_slot_combos(self) -> list[QtWidgets.QComboBox]:
        return (
            list(self.player_team_combos["combos"])
            + list(self.opponent_team_combos["combos"])
            + list(self.player_active_combos["combos"])
            + list(self.opponent_active_combos["combos"])
            + [self.damage_attacker, self.damage_defender]
        )

    def _format_changed(self) -> None:
        if self._syncing:
            return
        self.snapshot = self.snapshot.with_format(BattleFormat.parse(self.format_combo.currentData()))
        self.apply_manual_selection()

    def _toggle_always_on_top(self, checked: bool) -> None:
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, checked)
        self.show()


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, config: AppConfig, config_path: Path) -> None:
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.setWindowTitle("ROI Calibration")
        self.resize(920, 720)
        self.inputs: dict[str, tuple[QtWidgets.QSpinBox, QtWidgets.QSpinBox, QtWidgets.QSpinBox, QtWidgets.QSpinBox]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel("填写截图区域坐标后保存。建议先用 capture 保存截图，再用图片查看器确认 x/y/width/height。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        for key in ROI_KEYS:
            rect = self.config.rois[key]
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            boxes = []
            for label, value in (("x", rect.x), ("y", rect.y), ("w", rect.width), ("h", rect.height)):
                row_layout.addWidget(QtWidgets.QLabel(label))
                box = QtWidgets.QSpinBox()
                box.setRange(0, 10000)
                box.setValue(value)
                row_layout.addWidget(box)
                boxes.append(box)
            form.addRow(key, row)
            self.inputs[key] = tuple(boxes)  # type: ignore[assignment]

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self) -> None:
        for key, boxes in self.inputs.items():
            x, y, width, height = (box.value() for box in boxes)
            self.config.rois[key] = Rect(x=x, y=y, width=width, height=height)
        save_config(self.config, self.config_path)
        self.accept()


def run_app(config: AppConfig) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(config)
    window.show()
    return int(app.exec())


def run_calibration(config: AppConfig, config_path: Path) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = CalibrationDialog(config, config_path)
    return 0 if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted else 1
