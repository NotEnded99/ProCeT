"""
读取 nr_results_v8/ 下所有系统和激活函数的验证结果，
生成汇总表格，保存到 nr_results_v8/ 目录下。

v9 指标说明:
    orig_h: 原始最大深度验证的 HarmonicMeanPassRate
    orig_std: 原始最大深度验证的 standard_pass_rate
    orig_R_s: 原始最大深度验证的 R_safe
    orig_R_u: 原始最大深度验证的 R_unsafe
    final_h: 最终 HarmonicMeanPassRate
    final_std: 最终 standard_pass_rate
    final_R_s: 最终 R_safe
    final_R_u: 最终 R_unsafe
    h_imp: HarmonicMeanPassRate 提升量 (final_h - orig_h)
    std_imp: standard_pass_rate 提升量 (final_std - orig_std)
"""

import os
import json
import glob

# 所有系统和激活函数组合
DYNAMICS_SYSTEMS = ['simple2d', 'barr1', 'barr2', 'barr3', 'barr4', "cartpole"] # 'simple_2d'
ACTIVATIONS = ['LeakyRelu', 'Tanh', 'Sigmoid']


def load_all_results(results_dir):
    """读取所有 JSON 结果文件，返回汇总列表。"""
    results = []

    for system in DYNAMICS_SYSTEMS:
        for activation in ACTIVATIONS:
            # filename = f"result_{system}_{activation}_v9.json"
            # filename = f"result_{system}_{activation}_v8_clean.json"
            # filename = f"result_{system}_{activation}_compare_v10.json"
            # filename = f"result_{system}_{activation}_v11_lbp.json"
            filename = f"result_{system}_{activation}_v12_lbp_w.json"
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
    headers = ['System', 'Act', 'orig_h(%)', 'orig_std(%)', 'orig_R_s(%)', 'orig_R_u(%)',
               'final_h(%)', 'final_std(%)', 'final_R_s(%)', 'final_R_u(%)',
               'h_imp(%)', 'std_imp(%)', 'Iters', 'Timestamp']

    rows = []
    for r in results:
        if not r.get('found', True):
            rows.append([r['system'], r['activation'], 'N/A', 'N/A', 'N/A', 'N/A',
                         'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
            continue

        orig_h = r.get('original_max_depth_harmonic', 0)
        orig_std = r.get('original_max_depth_standard', 0)
        orig_R_s = r.get('original_max_depth_R_safe', 0)
        orig_R_u = r.get('original_max_depth_R_unsafe', 0)
        final_h = r.get('final_harmonic_pass_rate', 0)
        final_std = r.get('final_standard_pass_rate', 0)
        final_R_s = r.get('final_R_safe', 0)
        final_R_u = r.get('final_R_unsafe', 0)
        h_imp = r.get('harmonic_improvement', 0)
        std_imp = r.get('standard_improvement', 0)
        iters = r.get('num_iterations', 0)
        ts = r.get('timestamp', 'N/A')

        rows.append([
            r['system'],
            r['activation'],
            f"{orig_h:.2f}",
            f"{orig_std:.2f}",
            f"{orig_R_s:.2f}",
            f"{orig_R_u:.2f}",
            f"{final_h:.2f}",
            f"{final_std:.2f}",
            f"{final_R_s:.2f}",
            f"{final_R_u:.2f}",
            f"{h_imp:+.2f}",
            f"{std_imp:+.2f}",
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
    lines.append("# Neural CBF Repair Results Summary \n")

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
    # results_dir = os.path.join(script_dir, "nr_results_v9")
    # results_dir = os.path.join(script_dir, "nr_results_compare_v10")
    # results_dir = os.path.join(script_dir, "nr_results_v11_lbp")
    results_dir = os.path.join(script_dir, "nr_results_v12_lbp_w")


    
    # 


    if not os.path.exists(results_dir):
        print(f"目录不存在: {results_dir}")
        print("请先运行 main_v8.py 生成结果。")
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
    base = os.path.join(results_dir, "results_summary_v9")
    save_table_markdown(headers, rows, base + ".md")
    save_table_csv(headers, rows, base + ".csv")
    save_summary_json(headers, rows, base + ".json")

    # 额外统计
    print("\n" + "─" * 60)
    print("统计摘要")
    print("─" * 60)

    valid_results = [r for r in results if r.get('found', True)]
    if valid_results:
        avg_h_imp = sum(r.get('harmonic_improvement', 0) for r in valid_results) / len(valid_results)
        max_h_imp = max(r.get('harmonic_improvement', 0) for r in valid_results)
        min_h_imp = min(r.get('harmonic_improvement', 0) for r in valid_results)

        avg_std_imp = sum(r.get('standard_improvement', 0) for r in valid_results) / len(valid_results)

        final_h_rates = [r.get('final_harmonic_pass_rate', 0) for r in valid_results]
        max_h_rate = max(final_h_rates)
        min_h_rate = min(final_h_rates)

        print(f"  HarmonicMeanPassRate 平均提升: {avg_h_imp:+.2f}%")
        print(f"  HarmonicMeanPassRate 最大提升: {max_h_imp:+.2f}%")
        print(f"  HarmonicMeanPassRate 最小提升: {min_h_imp:+.2f}%")
        print(f"  HarmonicMeanPassRate 最终范围: {min_h_rate:.2f}% ~ {max_h_rate:.2f}%")
        print(f"  standard_pass_rate 平均提升: {avg_std_imp:+.2f}%")
        print(f"  成功提升 HarmonicMeanPassRate 的组合数: {sum(1 for r in valid_results if r.get('harmonic_improvement', 0) > 0)}/{len(valid_results)}")


if __name__ == "__main__":
    main()
