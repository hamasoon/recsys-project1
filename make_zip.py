# 제출용 zip 생성: python make_zip.py
# PDF는 report/에 있지만 zip 루트에 넣는다. results/, cache/, input_*.txt는 제출 대상이 아님
import os
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
NAME = "PRJ1_202527567_배문성"
FILES = [                                          # (프로젝트 내 경로, zip 내 경로)
    (f"report/{NAME}.pdf", f"{NAME}.pdf"),
    ("main.py", "main.py"),
    ("validate.py", "validate.py"),
    ("requirements.txt", "requirements.txt"),
    ("data/games_metadata.json", "data/games_metadata.json"),
    ("data/ratings.csv", "data/ratings.csv"),
]

if __name__ == "__main__":
    missing = [src for src, _ in FILES if not os.path.isfile(os.path.join(BASE, src))]
    if missing:
        raise SystemExit(f"missing files: {missing}")

    out = os.path.join(BASE, f"{NAME}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in FILES:
            z.write(os.path.join(BASE, src), arc)

    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            print(f"{info.filename:40s} {info.file_size:>12,d}")
        assert z.testzip() is None
    print(f"-> {out} ({os.path.getsize(out):,d} bytes)")
