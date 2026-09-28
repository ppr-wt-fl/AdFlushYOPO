from adflush_encodings import *
import csv
import os
import requests
from multiprocessing import Pool, Process, TimeoutError
import time
import csv
import tldextract
from datetime import datetime
import sys
csv.field_size_limit(sys.maxsize)
import signal
import sqlite3
import glob
import plyvel

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")


ng_list = ["ng_0_0_0","ng_0_0_1","ng_0_0_2","ng_0_0_3","ng_0_0_5","ng_0_0_9","ng_0_0_10","ng_0_0_11","ng_0_0_12","ng_0_0_14","ng_0_0_15","ng_0_1_0","ng_0_1_1","ng_0_1_3","ng_0_1_15","ng_0_2_0","ng_0_2_1","ng_0_2_2","ng_0_2_3","ng_0_2_8","ng_0_2_10","ng_0_2_11","ng_0_2_15","ng_0_3_0","ng_0_3_1","ng_0_3_2","ng_0_3_3","ng_0_3_5","ng_0_3_9","ng_0_3_10","ng_0_3_11","ng_0_3_15","ng_0_5_0","ng_0_5_2","ng_0_5_3","ng_0_5_5","ng_0_5_9","ng_0_5_10","ng_0_5_15","ng_0_9_0","ng_0_9_3","ng_0_9_5","ng_0_9_10","ng_0_9_15","ng_0_10_0","ng_0_10_3","ng_0_10_15","ng_0_11_0","ng_0_11_2","ng_0_11_3","ng_0_11_11","ng_0_11_15","ng_0_12_3","ng_0_12_4","ng_0_12_5","ng_0_12_15","ng_0_14_0","ng_0_14_4","ng_0_15_0","ng_0_15_1","ng_0_15_2","ng_0_15_3","ng_0_15_5","ng_0_15_9","ng_0_15_10","ng_0_15_11","ng_0_15_12","ng_0_15_14","ng_0_15_15","ng_0_18_3","ng_0_18_14","ng_0_18_15","ng_1_0_0","ng_1_0_1","ng_1_0_2","ng_1_0_3","ng_1_0_5","ng_1_0_9","ng_1_0_10","ng_1_0_12","ng_1_0_15","ng_1_1_1","ng_1_1_3","ng_1_2_0","ng_1_2_4","ng_1_3_0","ng_1_3_1","ng_1_3_2","ng_1_3_3","ng_1_3_5","ng_1_3_9","ng_1_3_10","ng_1_3_11","ng_1_3_12","ng_1_3_15","ng_1_5_0","ng_1_5_1","ng_1_5_3","ng_1_5_5","ng_1_5_9","ng_1_5_10","ng_1_5_12","ng_1_5_15","ng_1_10_0","ng_1_10_15","ng_1_15_0","ng_1_15_1","ng_1_15_2","ng_1_15_3","ng_1_15_5","ng_1_15_10","ng_1_15_12","ng_1_15_15","ng_2_0_0","ng_2_0_1","ng_2_0_2","ng_2_0_3","ng_2_0_5","ng_2_0_9","ng_2_0_10","ng_2_0_12","ng_2_0_15","ng_2_0_18","ng_2_1_3","ng_2_2_0","ng_2_2_1","ng_2_2_2","ng_2_2_3","ng_2_2_4","ng_2_2_8","ng_2_2_10","ng_2_2_11","ng_2_2_13","ng_2_2_15","ng_2_3_0","ng_2_3_2","ng_2_3_3","ng_2_3_15","ng_2_4_0","ng_2_4_2","ng_2_4_3","ng_2_4_8","ng_2_4_10","ng_2_4_15","ng_2_10_0","ng_2_10_3","ng_2_10_15","ng_2_11_0","ng_2_11_1","ng_2_11_2","ng_2_11_3","ng_2_11_10","ng_2_11_11","ng_2_11_15","ng_2_13_0","ng_2_13_2","ng_2_13_3","ng_2_13_15","ng_2_15_0","ng_2_15_1","ng_2_15_2","ng_2_15_3","ng_2_15_5","ng_2_15_10","ng_2_15_11","ng_2_15_12","ng_2_15_14","ng_2_15_15","ng_3_0_0","ng_3_0_1","ng_3_0_2","ng_3_0_3","ng_3_0_5","ng_3_0_9","ng_3_0_10","ng_3_0_11","ng_3_0_12","ng_3_0_14","ng_3_0_15","ng_3_1_0","ng_3_1_1","ng_3_1_3","ng_3_2_0","ng_3_2_2","ng_3_2_3","ng_3_2_8","ng_3_2_11","ng_3_2_15","ng_3_3_0","ng_3_3_1","ng_3_3_2","ng_3_3_3","ng_3_3_9","ng_3_3_11","ng_3_3_15","ng_3_5_0","ng_3_5_1","ng_3_5_2","ng_3_5_3","ng_3_5_5","ng_3_5_9","ng_3_5_10","ng_3_5_15","ng_3_9_0","ng_3_9_3","ng_3_9_15","ng_3_10_15","ng_3_11_0","ng_3_11_1","ng_3_11_2","ng_3_11_3","ng_3_11_11","ng_3_11_15","ng_3_12_3","ng_3_12_4","ng_3_12_15","ng_3_14_4","ng_3_15_0","ng_3_15_1","ng_3_15_2","ng_3_15_3","ng_3_15_5","ng_3_15_9","ng_3_15_10","ng_3_15_12","ng_3_15_14","ng_3_15_15","ng_4_0_0","ng_4_0_1","ng_4_0_2","ng_4_0_3","ng_4_0_15","ng_4_1_3","ng_4_2_0","ng_4_2_2","ng_4_2_3","ng_4_2_4","ng_4_2_8","ng_4_2_11","ng_4_2_13","ng_4_2_15","ng_4_3_0","ng_4_3_2","ng_4_3_3","ng_4_3_15","ng_4_5_3","ng_4_5_15","ng_4_10_0","ng_4_10_3","ng_4_10_15","ng_4_11_2","ng_4_11_3","ng_4_11_15","ng_4_15_0","ng_4_15_1","ng_4_15_2","ng_4_15_3","ng_4_15_5","ng_4_15_10","ng_4_15_11","ng_4_15_12","ng_4_15_14","ng_4_15_15","ng_5_0_0","ng_5_0_1","ng_5_0_2","ng_5_0_3","ng_5_0_5","ng_5_0_12","ng_5_0_15","ng_5_1_0","ng_5_1_3","ng_5_1_15","ng_5_2_0","ng_5_2_4","ng_5_3_0","ng_5_3_2","ng_5_3_3","ng_5_3_5","ng_5_3_12","ng_5_3_15","ng_5_5_0","ng_5_5_3","ng_5_5_5","ng_5_5_9","ng_5_5_12","ng_5_5_15","ng_5_9_3","ng_5_9_5","ng_5_9_15","ng_5_10_0","ng_5_10_3","ng_5_10_15","ng_5_12_4","ng_5_15_0","ng_5_15_1","ng_5_15_2","ng_5_15_3","ng_5_15_5","ng_5_15_10","ng_5_15_11","ng_5_15_12","ng_5_15_15","ng_9_0_0","ng_9_0_1","ng_9_0_2","ng_9_0_3","ng_9_0_5","ng_9_0_9","ng_9_0_10","ng_9_0_12","ng_9_0_15","ng_9_3_0","ng_9_3_1","ng_9_3_2","ng_9_3_3","ng_9_3_15","ng_9_5_0","ng_9_5_2","ng_9_5_3","ng_9_5_5","ng_9_5_9","ng_9_5_10","ng_9_5_12","ng_9_5_15","ng_9_10_15","ng_9_15_0","ng_9_15_1","ng_9_15_2","ng_9_15_3","ng_9_15_5","ng_9_15_10","ng_9_15_12","ng_9_15_15","ng_10_0_0","ng_10_0_2","ng_10_0_3","ng_10_0_12","ng_10_0_15","ng_10_3_0","ng_10_3_1","ng_10_3_3","ng_10_3_15","ng_10_15_0","ng_10_15_1","ng_10_15_3","ng_10_15_15","ng_11_0_0","ng_11_0_2","ng_11_0_3","ng_11_0_15","ng_11_1_3","ng_11_2_0","ng_11_2_2","ng_11_2_3","ng_11_2_8","ng_11_2_10","ng_11_2_11","ng_11_2_15","ng_11_3_0","ng_11_3_1","ng_11_3_2","ng_11_3_3","ng_11_3_10","ng_11_3_11","ng_11_3_15","ng_11_10_15","ng_11_11_0","ng_11_11_2","ng_11_11_3","ng_11_11_15","ng_11_15_0","ng_11_15_1","ng_11_15_2","ng_11_15_3","ng_11_15_5","ng_11_15_10","ng_11_15_11","ng_11_15_12","ng_11_15_15","ng_12_3_3","ng_12_3_15","ng_12_4_0","ng_12_4_1","ng_12_4_2","ng_12_4_3","ng_12_4_5","ng_12_4_8","ng_12_4_10","ng_12_4_11","ng_12_4_15","ng_12_5_15","ng_12_15_0","ng_12_15_1","ng_12_15_2","ng_12_15_3","ng_12_15_5","ng_12_15_10","ng_12_15_12","ng_12_15_14","ng_12_15_15","ng_13_0_2","ng_13_0_3","ng_13_0_15","ng_13_2_0","ng_13_2_2","ng_13_2_3","ng_13_2_8","ng_13_2_10","ng_13_2_11","ng_13_2_15","ng_13_3_0","ng_13_3_2","ng_13_3_3","ng_13_3_15","ng_13_15_0","ng_13_15_2","ng_13_15_3","ng_13_15_12","ng_13_15_15","ng_14_0_0","ng_14_0_2","ng_14_0_3","ng_14_0_9","ng_14_0_12","ng_14_0_14","ng_14_0_15","ng_14_4_0","ng_14_4_2","ng_14_4_3","ng_14_4_8","ng_14_4_15","ng_15_0_0","ng_15_0_1","ng_15_0_2","ng_15_0_3","ng_15_0_5","ng_15_0_9","ng_15_0_10","ng_15_0_11","ng_15_0_12","ng_15_0_14","ng_15_0_15","ng_15_1_0","ng_15_1_1","ng_15_1_2","ng_15_1_3","ng_15_1_5","ng_15_1_10","ng_15_1_15","ng_15_2_0","ng_15_2_2","ng_15_2_3","ng_15_2_4","ng_15_2_8","ng_15_2_11","ng_15_2_13","ng_15_2_15","ng_15_3_0","ng_15_3_1","ng_15_3_2","ng_15_3_3","ng_15_3_5","ng_15_3_9","ng_15_3_10","ng_15_3_11","ng_15_3_12","ng_15_3_14","ng_15_3_15","ng_15_5_0","ng_15_5_1","ng_15_5_2","ng_15_5_3","ng_15_5_5","ng_15_5_9","ng_15_5_10","ng_15_5_12","ng_15_5_15","ng_15_9_0","ng_15_9_3","ng_15_9_5","ng_15_9_15","ng_15_10_0","ng_15_10_3","ng_15_10_15","ng_15_11_0","ng_15_11_2","ng_15_11_3","ng_15_11_15","ng_15_12_3","ng_15_12_4","ng_15_12_15","ng_15_14_0","ng_15_14_4","ng_15_15_0","ng_15_15_1","ng_15_15_2","ng_15_15_3","ng_15_15_5","ng_15_15_9","ng_15_15_10","ng_15_15_11","ng_15_15_12","ng_15_15_14","ng_15_15_15","ng_15_15_17","ng_15_17_0","ng_15_17_12","ng_15_17_15","ng_17_0_0","ng_17_0_2","ng_17_0_15","ng_17_12_4","ng_17_15_0","ng_17_15_15","ng_18_3_0","ng_18_3_2","ng_18_3_3","ng_18_3_15","ng_18_14_0","ng_18_14_4","ng_18_15_0","ng_18_15_1","ng_18_15_2","ng_18_15_3","ng_18_15_5","ng_18_15_10","ng_18_15_15"]

def extract_fqdn(url):
    extracted = tldextract.extract(url)
    if extracted.subdomain:
        return f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}"
    else:
        return f"{extracted.domain}.{extracted.suffix}"

def is_js_file(url,  timeout=10):
    try:
        # Send a HEAD request to get the headers without downloading the entire file
        response = requests.head(url, timeout=timeout)
        time.sleep(0.1)
        # Check if the content type header indicates JavaScript
        content_type = response.headers.get("content-type", "")
        return "javascript" in content_type.lower()
    except Exception as e:
        print(f"Error checking file type: {e}")
        return False

def save_js_file(url, filename):
    # Send a GET request to download the JavaScript file
    response = requests.get(url, timeout=60)
    with open(filename, "wb") as file:
        file.write(response.content)

def save_html_file(url, filename):
    # Send a GET request to download the JavaScript file
    response = requests.get(url, timeout=60)
    with open(filename, "wb") as file:
        file.write(response.content)

def set_dictionary_values(csv_file, dictionary):
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            for key, value in row.items():
                if key in dictionary:
                    dictionary[key] = value

def load_content_hashes(crawl_dir):
    hashes = {}
    for db_file in glob.glob(crawl_dir + "crawl-data_*.sqlite"):
        con = sqlite3.connect(db_file)
        for visit_id, url, content_hash in con.execute("SELECT visit_id, url, content_hash FROM http_responses WHERE content_hash IS NOT NULL AND content_hash != '<error>'"):
            hashes[(str(visit_id), url)] = content_hash
        con.close()
    return hashes

def dump_scripts(content_dir, content_hashes, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    for ldb_file in glob.glob(content_dir + "content_*.ldb"):
        db = plyvel.DB(ldb_file)
        for content_hash in content_hashes:
            path = save_directory + content_hash + ".js"
            if os.path.exists(path):
                continue
            body = db.get(content_hash.encode("ascii"))
            if body is not None:
                with open(path, "wb") as file:
                    file.write(body)
        db.close()

def pipeline(idx, start_row, end_row):
    save_directory = '/yopo-artifact/AdFlush/source/MY_jsfile/'
    csv_file_path = "/yopo-artifact/WebGraph/result_webgraph_unmod/merged_features_with_labelled_exclude_flow.csv"
    output_csv_file = "/yopo-artifact/data/dataset/from_adflush/features_raw_{}.csv".format(idx)

    tot_num = end_row - start_row
    print(f"This process handles {tot_num} rows")

    with open(csv_file_path, 'r') as file, open(output_csv_file, 'w', newline='') as csv_output:
        reader = csv.reader(file)
        writer = csv.writer(csv_output)

        header = next(reader)
        visit_id_idx = header.index("visit_id")
        new_header = header + ["brackettodot", "num_get_storage", "num_set_storage", "num_get_cookie", "num_requests_sent", "avg_ident", "avg_charperline"] + ['req_url_' + str(i) for i in range(0, 200)] + ['fqdn_' + str(i) for i in range(0, 30)] + ng_list
        writer.writerow(new_header)

        for index, row in enumerate(reader):
            if idx == 0 and index % 10000 == 0:
                print(f"now processing {str(index)} instances...")
            if start_row <= index <= end_row:
                try:
                    cps = row[58]
                    url = row[2]  # Assuming third column contains the URL
                    top_domain = row[0]
                    fqdn = extract_fqdn(top_domain)
                    # print(cps, url, top_domain, fqdn)
                except Exception as err:
                    print(err)
                    print("err1")
                    continue

                try:
                    fqdn_emb = char2vec_pretrained(fqdn, False)
                except Exception as err:
                    print(err)
                    print("err2")
                    continue
                try:
                    url_emb = char2vec_pretrained(url, True)
                except Exception as err:
                    print(err)
                    print("err3")
                    continue

                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%03d")
                js_name = f"{current_time}.js"

                # print(fqdn_emb)
                # print("---")
                # print(url_emb)

                # # extract JS-related features
                # js_name = fqdn
                # js_name = url.split("/")[-1]
                # js_name = js_name.split(",")[-1]

                if cps == "script":
                    if not os.path.exists(save_directory):
                        os.makedirs(save_directory)
                    content_hash = content_hashes.get((row[visit_id_idx], url))
                    if content_hash and os.path.exists(save_directory + content_hash + ".js"):
                        js_name = content_hash
                    else:
                        try:
                            save_js_file(url, save_directory + js_name + ".js")
                        except Exception as e:
                            continue
                    try:
                        ast_depth, ast_breadth, avg_ident, avg_charperline, brackettodot, num_requests_sent, num_set_storage, num_get_storage, num_get_cookie, ngram = extract_JS_Features_shine_with_timeout(file_name=js_name, _isHTML=False, timeout=180)
                    except:
                        ast_depth = 0
                        ast_breadth = 0
                        avg_ident = 0
                        avg_charperline = 0
                        brackettodot = 0
                        num_requests_sent = 0
                        num_set_storage = 0
                        num_get_storage = 0
                        num_get_cookie = 0
                        ngram = [0] * 528

                        # concate columns
                        row += [brackettodot]
                        row += [num_get_storage]
                        row += [num_set_storage]
                        row += [num_get_cookie]
                        row += [num_requests_sent]
                        row += [avg_ident]
                        row += [avg_charperline]

                        row += url_emb.tolist()
                        row += fqdn_emb.tolist()
                        row += ngram

                        writer.writerow(row)
                        continue

                    # print("\nNew features for processing/now.js\n\t", "ast_depth: ",ast_depth, "ast_breadth: ",ast_breadth, "avg_ident: ",avg_ident," avg_charperline: ", avg_charperline, "brackettodot: ",brackettodot, "num_requests_sent: ",num_requests_sent, "num_set_storage: ",num_set_storage, "num_get_storage: ",num_get_storage, "num_get_cookie: ",num_get_cookie, "ngram: ",ngram)

                    row += [brackettodot]
                    row += [num_get_storage]
                    row += [num_set_storage]
                    row += [num_get_cookie]
                    row += [num_requests_sent]
                    row += [avg_ident]
                    row += [avg_charperline]

                    row += url_emb.tolist()
                    row += fqdn_emb.tolist()
                    row += [0]* 528

                    # print(ngram)
                    try:
                        for key, value in ngram.items():
                            # print(key, value)
                            if key in ngram:
                                try:
                                    column_index = new_header.index(key)
                                except:
                                    pass
                                try:
                                    row[column_index] = ngram[key]
                                except:
                                    continue
                    except:
                        continue
                
                else:
                    ast_depth = 0
                    ast_breadth = 0
                    avg_ident = 0
                    avg_charperline = 0
                    brackettodot = 0
                    num_requests_sent = 0
                    num_set_storage = 0
                    num_get_storage = 0
                    num_get_cookie = 0
                    ngram = [0] * 528

                    # concate columns
                    row += [brackettodot]
                    row += [num_get_storage]
                    row += [num_set_storage]
                    row += [num_get_cookie]
                    row += [num_requests_sent]
                    row += [avg_ident]
                    row += [avg_charperline]

                    row += url_emb.tolist()
                    row += fqdn_emb.tolist()
                    row += ngram

                
                writer.writerow(row)

processes = []
file_openwpm = "/yopo-artifact/WebGraph/result_webgraph_unmod/merged_features_with_labelled_exclude_flow.csv"
data_openwpm = pd.read_csv(file_openwpm)
num_total = len(data_openwpm)

# LevelDB allows one process per DB, so script bodies are dumped here before the workers fork
content_hashes = load_content_hashes("/yopo-artifact/OpenWPM/datadir_proxy_unmod/crawl_dir/")
scripts = data_openwpm[data_openwpm.iloc[:, 58] == "script"]
needed = {content_hashes[key] for key in zip(scripts["visit_id"].astype(str), scripts.iloc[:, 2]) if key in content_hashes}
print(f"{len(needed)} script bodies found in LevelDB for {len(scripts)} script rows")
dump_scripts("/yopo-artifact/OpenWPM/datadir_proxy_unmod/content_dir/", needed, '/yopo-artifact/AdFlush/source/MY_jsfile/')
num_core = 32
per_core = (num_total + num_core - 1) // num_core

for i in range(num_core):
    name = "Process" + str(i)
    start_idx = i * per_core
    end_idx = min((i + 1) * per_core, num_total)
    print(name)
    p = Process(target=pipeline, args=(i, start_idx, end_idx))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
