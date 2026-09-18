import os
import json
import re
import base64
import subprocess
import atexit
import shutil
import threading
import time

# Track generated files to clean them up later
generated_files = []
generated_dirs = []

def cleanup():
    print("\n[Polyglot] Cleaning up generated ghost files...")
    for f in generated_files:
        if os.path.exists(f):
            os.remove(f)
    for d in generated_dirs:
        if os.path.exists(d) and not os.listdir(d):
            os.rmdir(d)
    print("[Polyglot] Workspace is pristine.")

atexit.register(cleanup)

def build_tutorials():
    print("[Polyglot] Generating literate tutorials...")
    with open('zensical.toml', 'r') as f:
        config_text = f.read()
    
    match = re.search(r'polyglot_tutorials\s*=\s*\[(.*?)\]', config_text, re.DOTALL)
    if not match: return
    
    tutorials_raw = match.group(1).replace('"', '').replace("'", "").split(',')
    tutorials = [t.strip() for t in tutorials_raw if t.strip()]
    
    for tut_dir in tutorials:
        nb_path = os.path.join(tut_dir, 'main.ipynb')
        cpp_path = os.path.join(tut_dir, 'main.cpp')
        jl_path = os.path.join(tut_dir, 'main.jl')
        out_md_path = os.path.join(tut_dir, 'index.md')
        out_cpp_annotated = os.path.join(tut_dir, f'{os.path.basename(tut_dir)}.cpp')
        out_jl_annotated = os.path.join(tut_dir, f'{os.path.basename(tut_dir)}.jl')
        
        if not os.path.exists(nb_path): continue
        
        with open(nb_path, 'r') as f: nb = json.load(f)
        with open(cpp_path, 'r') as f: cpp_src = f.read()
        with open(jl_path, 'r') as f: jl_src = f.read()
        
        base_name = os.path.basename(tut_dir)
        
        def extract_block(src, block_id, lang):
            pattern = f'// \\[ID: {block_id}\\](.*?)// \\[END: {block_id}\\]' if lang == 'cpp' else f'# \\[ID: {block_id}\\](.*?)# \\[END: {block_id}\\]'
            match = re.search(pattern, src, re.DOTALL)
            return match.group(1).strip() if match else f"// Error: ID {block_id} not found"
            
        md_output = []
        cpp_annotated = ["/*\n * QKRYLOV FULLY ANNOTATED TUTORIAL\n */\n"]
        jl_annotated = ["#=\n = QKRYLOV FULLY ANNOTATED TUTORIAL\n =#\n"]
        current_md_buffer = []
        
        img_dir = os.path.join(tut_dir, 'images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)
            generated_dirs.append(img_dir)
        
        for cell_idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                text = "".join(cell['source'])
                md_output.append(text)
                current_md_buffer.append(text)
            elif cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                
                text_outputs = []
                image_markdowns = []
                for out in cell.get('outputs', []):
                    if out['output_type'] == 'stream':
                        text_outputs.extend(out['text'])
                    elif out['output_type'] in ['execute_result', 'display_data']:
                        data = out.get('data', {})
                        if 'image/png' in data:
                            img_data = base64.b64decode(data['image/png'])
                            img_filename = f"plot_{cell_idx}.png"
                            img_path = os.path.join(img_dir, img_filename)
                            with open(img_path, "wb") as img_f:
                                img_f.write(img_data)
                            generated_files.append(img_path)
                            image_markdowns.append(f"\n![Output Image](images/{img_filename})\n")
                        elif 'text/plain' in data:
                            text_outputs.extend(data['text/plain'])
                
                formatted_text_output = ""
                if text_outputs:
                    out_str = "".join(text_outputs).strip()
                    formatted_text_output = "\n\n# Output:\n" + "\n".join(f"# {line}" for line in out_str.split('\n'))
                
                match = re.match(r'^#\s*\[ID:\s*(\w+)\]\s*\n?', source)
                if match:
                    block_id = match.group(1)
                    clean_source = source[match.end():]
                    py_code = clean_source + formatted_text_output
                    cpp_code = extract_block(cpp_src, block_id, 'cpp')
                    jl_code = extract_block(jl_src, block_id, 'jl')
                    
                    tabs = []
                    tabs.append(f'=== "Python"\n    ```python\n    ' + py_code.replace('\n', '\n    ') + '\n    ```\n')
                    tabs.append(f'=== "C++"\n    ```cpp\n    ' + cpp_code.replace('\n', '\n    ') + '\n    ```\n')
                    tabs.append(f'=== "Julia"\n    ```julia\n    ' + jl_code.replace('\n', '\n    ') + '\n    ```\n')
                    md_output.append("".join(tabs))
                    
                    md_text = "\n\n".join(current_md_buffer)
                    if md_text.strip():
                        cpp_annotated.append(f'/*\n{md_text}\n*/\n')
                        jl_annotated.append(f'#=\n{md_text}\n=#\n')
                    
                    cpp_annotated.append(cpp_code + "\n")
                    jl_annotated.append(jl_code + "\n")
                    current_md_buffer = [] 
                else:
                    py_code = source + formatted_text_output
                    md_output.append(f'```python\n{py_code}\n```')
                
                if image_markdowns:
                    md_output.extend(image_markdowns)
                    
        md_text = "\n\n".join(md_output)
        
        # Inject buttons
        cpp_dl_name = f"{base_name}.cpp"
        jl_dl_name = f"{base_name}.jl"
        
        cpp_logo = '<img src="https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg" width="18" style="vertical-align: middle; margin-bottom: 2px;">'
        jl_logo_dots = '<img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia-dots.svg" width="22" style="vertical-align: middle; margin-bottom: 2px;">'
        jupyter_logo = '<img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" width="18" style="vertical-align: middle; margin-bottom: 2px;">'
        
        nb_dl_name = f"{base_name}.ipynb"
        btn_cpp = f"[{cpp_logo} &nbsp;&nbsp; :material-download:]({cpp_dl_name} \"{cpp_dl_name}\"){{: .md-button .polyglot-btn }}"
        btn_jl = f"[{jl_logo_dots} &nbsp;&nbsp; :material-download:]({jl_dl_name} \"{jl_dl_name}\"){{: .md-button .polyglot-btn }}"
        btn_py = f"[{jupyter_logo} &nbsp;&nbsp; :material-download:]({nb_dl_name} \"{nb_dl_name}\"){{: .md-button .polyglot-btn }}"
        
        style_block = "<style>.polyglot-btn .twemoji { font-size: 1.5em !important; vertical-align: middle; }</style>"
        buttons = f"\n\n{style_block}\n{btn_py} &nbsp;&nbsp; {btn_jl} &nbsp;&nbsp; {btn_cpp}\n\n"
        md_text = re.sub(r'^(# .*?\n)', r'\1' + buttons, md_text, count=1, flags=re.MULTILINE)
        with open(out_md_path, 'w') as f: f.write(md_text)
        with open(out_cpp_annotated, 'w') as f: f.write("\n".join(cpp_annotated))
        with open(out_jl_annotated, 'w') as f: f.write("\n".join(jl_annotated))
        
        import copy
        clean_nb = copy.deepcopy(nb)
        for c in clean_nb.get('cells', []):
            if c.get('cell_type') == 'code':
                new_source = []
                for line in c.get('source', []):
                    if not re.match(r'^#\s*\[ID:\s*\w+\]\s*\n?', line):
                        new_source.append(line)
                c['source'] = new_source
        
        out_nb_annotated = os.path.join(tut_dir, nb_dl_name)
        with open(out_nb_annotated, 'w') as f: json.dump(clean_nb, f, indent=1)
        
        generated_files.extend([out_md_path, out_cpp_annotated, out_jl_annotated, out_nb_annotated])

def watch_files():
    import time, os, re
    with open('zensical.toml', 'r') as f:
        match = re.search(r'polyglot_tutorials\s*=\s*\[(.*?)\]', f.read(), re.DOTALL)
    if not match: return
    
    sources = []
    for tut in match.group(1).replace('"', '').replace("'", "").split(','):
        tut = tut.strip()
        if tut:
            sources.extend([os.path.join(tut, 'main.ipynb'), os.path.join(tut, 'main.cpp'), os.path.join(tut, 'main.jl')])
    
    last_mtime = {}
    for s in sources:
        if os.path.exists(s):
            last_mtime[s] = os.path.getmtime(s)
            
    while True:
        time.sleep(1.0)
        changed = False
        for s in sources:
            if os.path.exists(s):
                mtime = os.path.getmtime(s)
                if mtime > last_mtime.get(s, 0):
                    last_mtime[s] = mtime
                    changed = True
        if changed:
            print("\n[Polyglot Watcher] Source change detected. Regenerating ghost files...")
            build_tutorials()

if __name__ == "__main__":
    import sys
    build_tutorials()
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        print("[Polyglot] Starting Zensical Server...")
        threading.Thread(target=watch_files, daemon=True).start()
        try:
            import shutil
            if shutil.which("uv"):
                subprocess.run(["uv", "run", "zensical", "serve"])
            else:
                subprocess.run(["zensical", "serve"])
        except KeyboardInterrupt:
            pass
    elif len(sys.argv) > 1 and sys.argv[1] in ["build", "build_ci"]:
        print("[Polyglot] Building Zensical Site...")
        import shutil
        if sys.argv[1] == "build_ci" or not shutil.which("uv"):
            subprocess.run(["zensical", "build", "--clean"])
        else:
            subprocess.run(["uv", "run", "zensical", "build", "--clean"])

