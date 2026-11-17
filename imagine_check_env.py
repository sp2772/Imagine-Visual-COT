# Are all my imagine_*.py files from the same "drop"?  Run:  python imagine_check_env.py   (from the project folder)
import importlib, inspect, os, sys, time

def stamp(path): return f"{os.path.getsize(path):>7} bytes, modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))}"

def main():
    print("python", sys.version.split()[0], "| folder:", os.getcwd(), "\n")
    mods = {}
    for name in ("imagine_data", "imagine_adv_lib"):
        try: mods[name] = importlib.import_module(name); print(f"{name:18s} {mods[name].__file__}\n{'':18s} {stamp(mods[name].__file__)}")
        except Exception as e: print(f"{name}: FAILED to import ({type(e).__name__}: {e})")
    lib, data = mods.get("imagine_adv_lib"), mods.get("imagine_data")
    checks = []
    if lib:
        checks += [("lib: LIB_VERSION >= 4", getattr(lib, "LIB_VERSION", 0) >= 4), ("lib: suffix_of / trk_suffix / breakdown", all(hasattr(lib, n) for n in ("suffix_of", "trk_suffix", "breakdown"))),
                   ("lib: prepare_split(track=...)", "track" in inspect.signature(lib.prepare_split).parameters), ("lib: get_spatial_items (self-contained loader)", hasattr(lib, "get_spatial_items"))]
    if data:
        checks += [("data: CLIP loads safetensors (needed on torch<2.6)", "use_safetensors" in inspect.getsource(data.load_clip)),
                   ("data: get_spatial_cot(root=...) (only needed by old scripts)", "root" in inspect.signature(data.get_spatial_cot).parameters)]
    need = {"imagine_step8_advanced_a.py": ("spatial2", "--w_contrast", "breakdown"), "imagine_step8_advanced_b.py": ("LIB_VERSION", "--track"), "imagine_step8_advanced_c.py": ("LIB_VERSION", "--split"),
            "imagine_step8_advanced_a_eval.py": ("LIB_VERSION",), "imagine_step8_advanced_a0.py": ("caption_ce",), "imagine_build_spatial_v2.py": ("grid_walk", "test_ood"), "imagine_audit_data.py": ("GREEN-pixels",)}
    for fn, keys in need.items():
        if not os.path.exists(fn): checks.append((f"{fn}: present", False)); continue
        s = open(fn, encoding="utf-8").read(); checks.append((f"{fn}: up to date ({', '.join(keys)})", all(k in s for k in keys)))
    print()
    for label, ok in checks: print(f"  [{'OK ' if ok else 'OLD'}] {label}")
    bad = [l for l, ok in checks if not ok and "only needed by old" not in l]
    print("\nAll consistent." if not bad else f"\n{len(bad)} item(s) are outdated or missing -> copy the newest version of those files into this folder.")

if __name__ == "__main__":
    main()
