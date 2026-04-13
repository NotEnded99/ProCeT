"""
读取 nr_results_clean/ 下所有系统和激活函数的单次验证结果，
生成汇总表格，保存到 nr_results_clean/ 目录下。
"""

import os
import json
import glob

# 所有系统和激活函数组合
DYNAMICS_SYSTEMS = ['simple_2d', 'barr1', 'barr2', 'barr3']
ACTIVATIONS = ['Relu', 'Tanh', 'Sigmoid']


def load_all_results(results_dir):
    """读取所有 JSON 结果文件，返回汇总列表。"""
    results = []

    for system in DYNAMICS_SYSTEMS:
        for activation in ACTIVATIONS:
            filename = f"result_{system}_{activation}_v7.json"
            filepath = os.path.join(results_dir, filename)

            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['found'] = True
                    results.append(data)
            else:
                # 结果不存在，标记为缺失
                results.append({
                    'system': system,
                    'activation': activation,
                    'found': False,
                })

    return results


def build_table(results):
    """构建表格数据。"""
    headers = ['System', 'Activation', 'Initial (%)', 'Final (%)', 'Improvement (%)',
               'F_h', 'F_safe', 'F_depth', 'F_unsafe', 'Iterations', 'Timestamp']

    rows = []
    for r in results:
        if not r.get('found', True):
            rows.append([r['system'], r['activation'], 'N/A', 'N/A', 'N/A',
                         'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
            continue

        init_pct = r.get('initial_pass_rate', 0)
        final_pct = r.get('final_pass_rate', 0)
        improv = r.get('improvement', 0)
        iters = r.get('num_iterations', 0)
        ts = r.get('timestamp', 'N/A')

        ir = r.get('initial_regions', {})
        f_h_init = ir.get('F_h_positive_in_unsafe', 0)
        f_safe_init = ir.get('F_safe_cbf_violation', 0)
        f_depth_init = ir.get('F_depth_limit_reached', 0)
        f_unsafe_init = ir.get('F_unsafe_cannot_split', 0)

        rows.append([
            r['system'],
            r['activation'],
            f"{init_pct:.2f}",
            f"{final_pct:.2f}",
            f"{improv:+.2f}",
            str(f_h_init),
            str(f_safe_init),
            str(f_depth_init),
            str(f_unsafe_init),
            str(iters),
            ts,
        ])

    return headers, rows


def print_table(headers, rows):
    """打印 ASCII 表格到控制台。"""
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]

    # 表头
    header_line = "│ " + " │ ".join(
        headers[i].center(col_widths[i]) for i in range(len(headers))
    ) + " │"
    sep_line = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"

    print("\n" + "═" * len(header_line))
    print("Neural CBF Repair Results Summary")
    print("═" * len(header_line))
    print(sep_line.replace("─", "═").replace("┼", "╪"))
    print(header_line)
    print(sep_line)

    for row in rows:
        line = "│ " + " │ ".join(
            str(row[i]).center(col_widths[i]) for i in range(len(row))
        ) + " │"
        print(line)

    print(sep_line.replace("─", "═").replace("┼", "╪"))


def save_table_markdown(headers, rows, output_path):
    """保存为 Markdown 表格。"""
    lines = []
    lines.append("# Neural CBF Repair Results Summary\n")

    # 表头行
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\nMarkdown 表格已保存: {output_path}")


def save_table_csv(headers, rows, output_path):
    """保存为 CSV 表格。"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    print(f"CSV 表格已保存: {output_path}")


def save_summary_json(headers, rows, output_path):
    """保存为汇总 JSON。"""
    summary = []
    for row in rows:
        entry = {headers[i]: row[i] for i in range(len(headers))}
        summary.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"汇总 JSON 已保存: {output_path}")


def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "nr_results_v7")

    if not os.path.exists(results_dir):
        print(f"目录不存在: {results_dir}")
        print("请先运行 main_clean.py 生成结果。")
        return

    print(f"读取结果目录: {results_dir}\n")

    # 读取所有结果
    results = load_all_results(results_dir)

    # 检查有多少结果存在
    found_count = sum(1 for r in results if r.get('found', True))
    total_count = len(results)
    print(f"找到 {found_count}/{total_count} 个结果文件")

    if found_count == 0:
        print("没有找到任何结果文件！")
        return

    # 构建表格
    headers, rows = build_table(results)

    # 打印到控制台
    print_table(headers, rows)

    # 保存文件
    base = os.path.join(results_dir, "results_summary")
    save_table_markdown(headers, rows, base + ".md")
    save_table_csv(headers, rows, base + ".csv")
    save_summary_json(headers, rows, base + ".json")

    # 额外统计
    print("\n" + "─" * 60)
    print("统计摘要")
    print("─" * 60)

    valid_results = [r for r in results if r.get('found', True)]
    if valid_results:
        avg_improvement = sum(r.get('improvement', 0) for r in valid_results) / len(valid_results)
        max_improvement = max(r.get('improvement', 0) for r in valid_results)
        min_improvement = min(r.get('improvement', 0) for r in valid_results)

        final_rates = [r.get('final_pass_rate', 0) for r in valid_results]
        max_rate = max(final_rates)
        min_rate = min(final_rates)

        print(f"  平均通过率提升: {avg_improvement:+.2f}%")
        print(f"  最大提升: {max_improvement:+.2f}%")
        print(f"  最小提升: {min_improvement:+.2f}%")
        print(f"  最终通过率范围: {min_rate:.2f}% ~ {max_rate:.2f}%")
        print(f"  成功修复的组合数: {sum(1 for r in valid_results if r.get('improvement', 0) > 0)}/{len(valid_results)}")


if __name__ == "__main__":
    main()
