import pandas as pd
import re
import requests

# 1. 加载原始数据
df = pd.read_csv('data/TPDdb/PROTAC_activity.txt', sep='\t')


def get_num(s):
    """提取数值、单位及操作符（如 <, >）"""
    if pd.isna(s):
        return None, None, None
    s = str(s).replace(',', '')
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if not nums:
        return None, None, None
    num = float(nums[0])
    unit = 'nM'
    if 'um' in s.lower() or 'µm' in s.lower():
        unit = 'uM'
    elif '%' in s:
        unit = '%'
    op = '<' if '<' in s else ('>' if '>' in s else None)
    return num, unit, op

def categorize_stan_strict(row):
    """
    参考 stan (2025) 论文的严格标注逻辑:
    - Active (1): DC50 <= 100nM 或 Dmax >= 80% 或 顶级定性等级 (A/+++)
    - Inactive (0): 其他所有
    """
    act_type = str(row['Activity Type']).lower()
    val = str(row['Activity']).strip()
    
    if pd.isna(row['Activity']) or val == '.' or val == '':
        return None

    # A. 定性等级映射 (stan 仅视顶级为 Active)
    if val in ['A', '+++', '++++']: return 1
    if val in ['B', '++', 'C', '+', 'D', '-', 'No']: return 0

    # B. 数值提取与单位标准化
    num, unit, op = get_num(val)
    if num is None: return None
    if unit == 'uM': num *= 1000 # 统一转为 nM
    
    # C. 基于 stan 阈值的严格判定
    # 浓度类指标 (DC50, IC50, EC50等)
    if any(k in act_type for k in ['dc50', 'ic50', 'gi50', 'ec50']):
        if op == '>': 
            return 0 # 大于任何值在此标准下通常都不属于强效
        if op == '<':
            return 1 if num <= 100 else 0
        return 1 if num <= 100 else 0
    
    # 降解深度指标 (Dmax, % Degradation)
    if 'dmax' in act_type or 'degradation' in act_type or unit == '%':
        return 1 if num >= 80 else 0
        
    return None

# 执行标注
df['Label'] = df.apply(categorize_stan_strict, axis=1)

# 剔除无法解析的无效数据 (约 1.5k 条)
final_df = df.dropna(subset=['Label']).copy()
final_df['Label'] = final_df['Label'].astype(int)

"""
后处理部分：
- 使用 mygene.info 为 Ligase 名称补充 UniProt ID（Ligase Uniprot 列）
- 对 Target ID / Ligase Uniprot 中的多 UniProt 条目进行拆分、去重
- 结合每行的 Ligase Uniprot，将混合的 POI/E3 ID 拆成干净的一对多、多对多关系
- 将多蛋白记录按 (POI_Uniprot, Ligase_Uniprot_clean) 展开为多条记录
"""

# 读取主表，按 TPD ID 进行关联
main_df = pd.read_csv('data/TPDdb/PROTAC_main_table.txt', sep='\t')
merged_df = final_df.merge(main_df, on='TPD ID', how='left')


# 一些常见 E3 ligase 的人工映射（覆盖/补充自动查询）
LIGASE_UNIPROT_MANUAL = {
    "VHL": "P40337",
    "CRBN": "Q96SW2",
    "IAP": "Q13490",
    "cIAP1": "Q13490",
    "Keap1": "Q14145",
    "KEAP1": "Q14145",
    "XIAP": "P98170",
    "MDM2": "Q00987",
    "UBR box": "Q8N806",
    "FEM1B": "Q9UK73",
    "DCAF1": "Q9Y4B6",
    "DCAF16": "Q9NXF7",
}


def get_uniprot_id_for_ligase(ligase_name: str):
    """
    调用 mygene.info 接口，通过 E3 ligase 名称查询 UniProt ID。
    逻辑与 clean.ipynb 中的实现保持一致：
    - 优先返回 Swiss-Prot，其次 TrEMBL
    - 统一返回一个字符串 ID，查不到则返回 None
    """
    if pd.isna(ligase_name):
        return None

    # 先查人工映射表，覆盖大小写完全一致的 key
    key = str(ligase_name)
    if key in LIGASE_UNIPROT_MANUAL:
        return LIGASE_UNIPROT_MANUAL[key]

    url = "https://mygene.info/v3/query"
    params = {
        "q": ligase_name,
        "species": "human",
        "fields": "uniprot",
        "size": 1,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        if not hits:
            return None

        hit = hits[0]
        uniprot = hit.get("uniprot")
        if not uniprot:
            return None

        # uniprot 可能是 dict / str / list
        if isinstance(uniprot, dict):
            if "Swiss-Prot" in uniprot:
                return (
                    uniprot["Swiss-Prot"][0]
                    if isinstance(uniprot["Swiss-Prot"], list)
                    else uniprot["Swiss-Prot"]
                )
            if "TrEMBL" in uniprot:
                return (
                    uniprot["TrEMBL"][0]
                    if isinstance(uniprot["TrEMBL"], list)
                    else uniprot["TrEMBL"]
                )
        elif isinstance(uniprot, str):
            return uniprot
        elif isinstance(uniprot, list):
            return uniprot[0] if uniprot else None
        return None
    except Exception as e:
        print(f"Error querying UniProt for '{ligase_name}': {e}")
        return None


# 基于合并后的表，为 Ligase 补充 UniProt ID（如果还没有该列）
if "Ligase Uniprot" not in merged_df.columns:
    if "Ligase" in merged_df.columns:
        ligase_names = merged_df["Ligase"].dropna().unique()
        ligase_uniprot_dict = {}
        for name in ligase_names:
            ligase_uniprot_dict[name] = get_uniprot_id_for_ligase(name)

        merged_df["Ligase Uniprot"] = merged_df["Ligase"].map(ligase_uniprot_dict)
    else:
        print("警告：在合并后的表中未找到 'Ligase' 列，无法添加 Ligase Uniprot。")


# ========== 仅按 Target ID 拆分多 UniProt 条目 ==========
# 允许所有以大写字母开头的标准 UniProt accession（6 位：字母 + 数字 + 3 字符 + 数字）
uniprot_pattern = re.compile(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$")


def parse_uniprot_list(s: str):
    """
    将一个可能包含多个 UniProt ID 的 Target ID 字符串拆成干净列表：
    - 按 / ; , 空格 等分隔
    - 去除空字符串
    - 只保留符合 UniProt accession 粗略正则的条目
    - 去重并排序
    """
    if pd.isna(s):
        return []
    tokens = re.split(r"[\/;, ]+", str(s))
    tokens = [t.strip() for t in tokens if t and t.strip()]

    # 按出现顺序去重，保留原始顺序，避免打乱 “前面 POI，后面 E3” 的语义
    seen = set()
    ids = []
    for t in tokens:
        if uniprot_pattern.match(t) and t not in seen:
            seen.add(t)
            ids.append(t)
    return ids


def explode_target_ids(row):
    """
    对单行按 Target ID 中的多 UniProt 条目拆分成多条：
    - 若 Target ID 拆分得到多个 ID，则为每个 ID 复制一条记录，并更新 Target ID 字段
    - 若拆分结果为空，则保留原始行（不会丢失数据）
    """
    ids = parse_uniprot_list(row.get("Target ID"))
    base = row.to_dict()

    # 没有解析出合法 UniProt，则原样返回一条
    if not ids:
        return [base]

    ligase_val = base.get("Ligase")
    has_ligase = isinstance(ligase_val, str) and ligase_val.strip() != ""

    # 情形一：Target ID 里同时写了 POI/E3（例如 P10275/Q12834），且 Ligase 为空
    #         按“前面 POI，后面 E3”的约定，拆成一条 (POI, E3) 记录
    if len(ids) >= 2 and not has_ligase:
        sym = str(base.get("Target Symbol") or "")
        sym_tokens = [t.strip() for t in re.split(r"[;/]", sym) if t.strip()]

        if len(sym_tokens) >= 2:
            poi_id, e3_id = ids[0], ids[1]
            poi_sym, e3_sym = sym_tokens[0], sym_tokens[1]

            rec = base.copy()
            # POI 使用第一个 ID/符号
            rec["Target ID"] = poi_id
            if "Target IDs" in rec:
                rec["Target IDs"] = poi_id
            rec["Target Symbol"] = poi_sym

            # E3 使用第二个 ID/符号，填入 Ligase 相关字段
            rec["Ligase"] = e3_sym
            rec["Ligase Uniprot"] = e3_id
            return [rec]

    # 情形二：普通多 POI（或已经有 Ligase），对每个 ID 拆成一条记录
    records = []
    for uid in ids:
        rec = base.copy()
        rec["Target ID"] = uid
        if "Target IDs" in rec:
            rec["Target IDs"] = uid
        records.append(rec)
    return records


expanded_records = []
for _, r in merged_df.iterrows():
    recs = explode_target_ids(r)
    expanded_records.extend(recs)

clean_df = pd.DataFrame(expanded_records)


# 保存文件：输出已经拆分好 Target ID 的版本
clean_df.to_csv("data/TPDdb/protac_label_with_main.csv", index=False)

print(f"处理完成，标注条数: {len(clean_df)}")
print(clean_df["Label"].value_counts())