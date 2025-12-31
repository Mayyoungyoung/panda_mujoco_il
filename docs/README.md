演示视频目录

请将演示视频放在此目录，命名为 `ep3.mp4`，然后执行：

```powershell
git add docs/ep3.mp4 README.md
git commit -m "Add demo video ep3.mp4 and README reference"
git push -u origin $(git branch --show-current)
```

说明：如果视频较大，建议使用 Git LFS 或外部托管（Release/Drive/S3）。