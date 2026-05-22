#!/usr/bin/env python3
"""
12口弹珠机最佳出卡策略 — 移动端 Web 界面
Mobile-friendly web interface for the 12-slot pinball strategy advisor.

Usage:
    pip install flask
    python web_app.py
    # 手机浏览器访问 http://<电脑局域网IP>:5000
"""

from flask import Flask, render_template, request, jsonify, session
from pinball_strategy import (
    NUM_SLOTS,
    MIN_BET,
    MAX_BET,
    DEFAULT_MULTIPLIER_SLOTS,
    PinballStrategy,
)
import os
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# 用 dict 存储每个 session 的 strategy 实例（简单方案，适合个人使用）
_strategies: dict = {}


def _get_strategy() -> PinballStrategy | None:
    sid = session.get("sid")
    if sid and sid in _strategies:
        return _strategies[sid]
    return None


def _set_strategy(strategy: PinballStrategy) -> None:
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_hex(16)
        session["sid"] = sid
    _strategies[sid] = strategy


@app.route("/")
def index():
    return render_template("index.html", NUM_SLOTS=NUM_SLOTS)


@app.route("/api/init", methods=["POST"])
def api_init():
    """初始化机器参数，创建策略实例"""
    data = request.get_json(force=True)
    try:
        # New tiered card rules (optional)
        card_tiers = data.get("card_tiers")
        if card_tiers:
            # Parse custom card tiers: [[threshold, cards], ...]
            card_tiers = [(int(t[0]), int(t[1])) for t in card_tiers]
            # Validate
            for t in card_tiers:
                if t[0] < 1 or t[1] < 1:
                    return jsonify({"error": "card_tiers 阈值和卡数必须 >= 1"}), 400
            # Sort by threshold
            card_tiers = sorted(card_tiers, key=lambda x: x[0])
        
        # Legacy T/J parameters (used if card_tiers not provided)
        T = int(data.get("T", 20))
        J = int(data.get("J", 10))
        
        priority = data.get("priority", "cards")
        prior_weight = float(data.get("prior_weight", 24))
        confidence_threshold = float(data.get("confidence_threshold", 5))
        current_marbles = int(data.get("current_marbles", 999))
        max_bet = int(data.get("max_bet", MAX_BET))

        if priority not in ("cards", "marbles"):
            return jsonify({"error": "priority 必须为 cards 或 marbles"}), 400
        if confidence_threshold < 0:
            return jsonify({"error": "信心阈值不能为负数"}), 400
        if current_marbles < 0:
            return jsonify({"error": "当前珠子数不能为负数"}), 400
        if not (MIN_BET <= max_bet <= MAX_BET):
            return jsonify({"error": f"最大投珠数必须在 {MIN_BET}~{MAX_BET} 之间"}), 400

        strategy = PinballStrategy(
            T=T, J=J, priority=priority,
            prior_weight=prior_weight, confidence_threshold=confidence_threshold,
            max_bet=max_bet, card_tiers=card_tiers, current_marbles=current_marbles,
        )
        _set_strategy(strategy)

        # 返回基准 EV 表
        ev_table = PinballStrategy.expected_value_table(strategy.T, strategy.J)
        return jsonify({
            "ok": True, 
            "ev_table": ev_table, 
            "T": strategy.T, 
            "J": strategy.J,
            "card_tiers": [[t[0], t[1]] for t in strategy.card_tiers],
            "priority": priority,
            "current_marbles": current_marbles,
        })
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """获取本局策略建议"""
    strategy = _get_strategy()
    if not strategy:
        return jsonify({"error": "请先设置机器参数"}), 400

    data = request.get_json(force=True)
    try:
        multiplier = int(data.get("multiplier", 2))
        lit_slots = [int(s) for s in data.get("lit_slots", [])]

        # 验证
        if multiplier not in DEFAULT_MULTIPLIER_SLOTS:
            return jsonify({"error": f"无效倍数: {multiplier}"}), 400
        expected_lit = DEFAULT_MULTIPLIER_SLOTS[multiplier]
        if len(lit_slots) != expected_lit:
            return jsonify({"error": f"{multiplier}x 倍数需要 {expected_lit} 个亮灯槽位"}), 400
        for s in lit_slots:
            if not (0 <= s < NUM_SLOTS):
                return jsonify({"error": f"槽位编号应在 0~{NUM_SLOTS-1} 之间"}), 400

        rec = strategy.recommend(multiplier, lit_slots)
        return jsonify({"ok": True, "recommendation": rec, "priority": strategy.priority})
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/record", methods=["POST"])
def api_record():
    """记录落点"""
    strategy = _get_strategy()
    if not strategy:
        return jsonify({"error": "请先设置机器参数"}), 400

    data = request.get_json(force=True)
    try:
        slot = int(data.get("slot", -1))
        if not (0 <= slot < NUM_SLOTS):
            return jsonify({"error": f"槽位编号应在 0~{NUM_SLOTS-1} 之间"}), 400
        strategy.record_landing(slot)
        return jsonify({
            "ok": True,
            "total_plays": strategy.total_plays,
            "landing_probs": [round(p, 4) for p in strategy.get_landing_probs()],
        })
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/probs", methods=["GET"])
def api_probs():
    """获取当前落点概率"""
    strategy = _get_strategy()
    if not strategy:
        return jsonify({"error": "请先设置机器参数"}), 400
    return jsonify({
        "ok": True,
        "total_plays": strategy.total_plays,
        "landing_probs": [round(p, 4) for p in strategy.get_landing_probs()],
    })


@app.route("/api/update_marbles", methods=["POST"])
def api_update_marbles():
    """更新当前珠子数量"""
    strategy = _get_strategy()
    if not strategy:
        return jsonify({"error": "请先设置机器参数"}), 400
    
    data = request.get_json(force=True)
    try:
        delta = int(data.get("delta", 0))
        strategy.update_marbles(delta)
        return jsonify({
            "ok": True,
            "current_marbles": strategy.current_marbles,
        })
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/set_marbles", methods=["POST"])
def api_set_marbles():
    """直接设置当前珠子数量"""
    strategy = _get_strategy()
    if not strategy:
        return jsonify({"error": "请先设置机器参数"}), 400
    
    data = request.get_json(force=True)
    try:
        count = int(data.get("count", 0))
        if count < 0:
            return jsonify({"error": "珠子数量不能为负数"}), 400
        strategy.current_marbles = count
        return jsonify({
            "ok": True,
            "current_marbles": strategy.current_marbles,
        })
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/analyze", methods=["GET"])
def api_analyze():
    """获取机器分析报告"""
    strategy = _get_strategy()
    if not strategy:
        return jsonify({"error": "请先设置机器参数"}), 400
    
    analysis = strategy.get_machine_analysis()
    return jsonify({
        "ok": True,
        "analysis": analysis,
    })


if __name__ == "__main__":
    import socket
    # 获取本机局域网 IP
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
    print(f"\n{'='*50}")
    print(f"  弹珠机策略顾问 — Web 版")
    print(f"  本机访问: http://127.0.0.1:5000")
    print(f"  手机访问: http://{local_ip}:5000")
    print(f"  (确保手机与电脑在同一 WiFi 网络下)")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
