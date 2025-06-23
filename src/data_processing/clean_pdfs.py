"""
Fix files and unify in a single format
"""

import os
import shutil


"""
Classify each report into respective year.pdf based on regular patterns
"""

base_years = list(range(2023,2013,-1)) # Usually reports are in 2014-2015 format, so we first try to match 2015
base_years_trim = list(range(23, 13, -1))
base_years += base_years_trim
patterns = []
for year in base_years:
    patterns.append(str(year)+".pdf")
    patterns.append(str(year)+"_en")
    patterns.append(str(year)+"-en")

print(patterns)
BASE_DIR = os.path.join("data", "reports")
firms = os.listdir(BASE_DIR)

pattern_files = {}
remaining_files = []
check_firms = []
total_files = 0
for firm in firms:
    if os.path.isdir(os.path.join(BASE_DIR, firm)):
        if "E.ON" not in firm:
            firm_name = firm.split("_")[0].split(".")[-1]
        else:
            firm_name = "EON"

        for year in os.listdir(os.path.join(BASE_DIR, firm)):
            file_path = os.path.join(BASE_DIR, firm, year)
            file_path = file_path.lower()
            if file_path.endswith(".pdf") or file_path.endswith(".PDF"):
                # Check for regular patters - 714 files
                check = 0
                for pattern in patterns:
                    if pattern in file_path:
                        pattern = pattern.split("_")[0].split(".")[0].split("-")[0]
                        if len(pattern) == 2:
                            pattern = "20" + pattern
                        pattern_files[file_path] = pattern
                        check = 1
                        break

                # Check for individual years - 955 files
                year_splits = year.split("-")
                year_splits.reverse()
                for sp in year_splits:
                    sub_splits = sp.split("_")
                    sub_splits.reverse()
                    for ssp in sub_splits:
                        for y in base_years:
                            if str(y) in ssp:
                                if not check: # Local Check
                                    if len(str(y)) == 2:
                                        y="20"+str(y)
                                    pattern_files[file_path] = str(y)
                                    check = 1
                                    break

                # Check for remaining files
                if not check:
                    check_firms.append(firm)
                    remaining_files.append(file_path)

                total_files += 1

# Fix remaining files manually
# 1354 PDF files in total, 6 corrupted, fix manually
"""
['data/reports/auto1 group se/full_auto1_group_annual_report_en.pdf', 'data/reports/grenke ag/01d4eab9-cf54-49b9-a2d0-1b4b2e5e6011.pdf', 'data/reports/grenke ag/ba6fac9d33a24c12b970ecafbd39fc5c.pdf', 'data/reports/nagarro se/nagarro_se_ar_en.pdf', 'data/reports/delivery hero/final_secured_en.pdf', 'data/reports/redcare pharmacy/ar-shop-apotheke-europe.pdf']
"""
assert len(pattern_files) + len(remaining_files) == total_files
print(total_files, len(remaining_files), len(pattern_files))

for k,v in pattern_files.items():
    if len(v) != 4:
        print(int(v))

# print(remaining_files)

# # Rename PDFs
# NEW_PTH = os.path.join("data", "cleaned_reports")
# os.makedirs(NEW_PTH, exist_ok=True)
#
# for k,v in pattern_files.items():
#     path_splits = os.path.split(k)
#     firm_name = path_splits[-2]
#     new_dir = os.path.join(NEW_PTH, firm_name)
#     os.makedirs(new_dir, exist_ok=True)
#     new_path = os.path.join(new_dir, v+".pdf")
#     if not os.path.exists(new_path):
#         shutil.copy(k, new_path)
#     else:
#         print(f"File {new_path} already exists!")


for k,v in pattern_files.items():
    print(k, v)
