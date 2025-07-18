
import os

import pandas as pd


def get_missing_files(company_dir):
    missing_files = []
    years = list(range(2014,2024))
    result_paths = [os.path.join(company_dir, str(year)+".pdf") for year in years]
    for pth in result_paths:
        if not os.path.exists(pth):
            missing_files.append(pth)

    return missing_files

def get_filtered_files(company_dir_texts, company_dir_pdfs):
    filtered_file_years = []
    years = list(range(2014,2024))

    for year in years:
        # replace with embedding_filter.json, to get first filtered data
        result_paths = os.path.join(company_dir_texts, str(year), "rag_filter_ai.json")
        pdf_path = os.path.join(company_dir_pdfs, str(year)+".pdf")
        if not os.path.exists(result_paths) and os.path.exists(pdf_path):
            filtered_file_years.append(year)

    return filtered_file_years


def generate_missing_files_list():
    companies = []
    years = []
    base_dir = os.path.join("data", "cleaned_reports")
    for company_dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, company_dir)):
            missing_files = get_missing_files(os.path.join(base_dir, company_dir))
            if missing_files:
                for file in missing_files:
                    splits = file.split("/")
                    companies.append(splits[-2])
                    years.append(int(splits[-1].replace(".pdf", "")))

    df = pd.DataFrame(
        {
            "Company": companies,
            "Year": years
        }
    )

    df.to_csv(os.path.join("results", "missing_data.csv"), index=False)

def generate_filtered_files_list():
    companies = []
    years = []
    base_dir = os.path.join("data", "cleaned_reports")
    for company_dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, company_dir)):
            company_dir_pdfs = os.path.join(base_dir, company_dir)
            company_dir_texts = company_dir_pdfs.replace("cleaned_reports", "texts")
            filtered_years = get_filtered_files(company_dir_texts, company_dir_pdfs)
            if filtered_years:
                for year in filtered_years:
                    companies.append(company_dir)
                    years.append(year)

    df = pd.DataFrame(
        {
            "Company": companies,
            "Year": years
        }
    )
    df.to_csv(os.path.join("results", "filtered_data_2.csv"), index=False)

if __name__ == '__main__':
    generate_missing_files_list()
    generate_filtered_files_list()