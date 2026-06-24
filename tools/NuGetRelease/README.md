# NuGetRelease — publishing Shorokoo to nuget.org

> **SUPERSEDED.** Releases are now cut by the `release` GitHub Actions
> workflow (MinVer tag-derived versions, Trusted Publishing — no stored API
> key) — see [`.github/workflows/release.yml`](../../.github/workflows/release.yml).
> This tool's manual version-stamping steps no longer match the MinVer setup;
> do not use it for a release.

An interactive console app that releases all Shorokoo packages. It asks for the
few things only you can provide (version number, API key, final go/no-go) and
automates the rest: version stamping, test run, packing, validation, and the
push in dependency order.

```bash
dotnet run --project tools/NuGetRelease              # full interactive release
dotnet run --project tools/NuGetRelease -- --dry-run # everything except the push
```

## One-time setup (you, once ever)

1. **Create a nuget.org account** at https://www.nuget.org (sign in with a
   Microsoft account).
2. **Reserve the `Shorokoo` ID prefix** (optional but recommended): on nuget.org,
   the first push of `Shorokoo` claims the ID; afterwards you can apply for
   [package ID prefix reservation](https://learn.microsoft.com/en-us/nuget/nuget-org/id-prefix-reservation)
   so only you can publish `Shorokoo.*` packages.
3. **Create an API key**: nuget.org → your account → *API Keys* → *Create*.
   - Glob pattern: `Shorokoo*`
   - Scopes: *Push new packages and package versions*
   - Store it in the `NUGET_API_KEY` environment variable, or paste it when the
     app asks (input is not echoed).

## What gets published

| Package | Contents |
|---|---|
| `Shorokoo` | Core framework |
| `Shorokoo.Modules` | Layers, losses, optimizers |
| `Shorokoo.CodeGen` | `[Module]` source generator (analyzer package) |
| `Shorokoo.OnnxRuntime` | Managed ONNX Runtime glue (dependency of the three below) |
| `Shorokoo.LinuxCPU` | Linux x64 CPU backend |
| `Shorokoo.LinuxGPU` | Linux x64 GPU (CUDA) backend |
| `Shorokoo.WinCPU` | Windows x64 CPU backend |
| `Shorokoo.WinGPU` | Windows x64 GPU (CUDA) backend |

Symbol packages (`.snupkg`) are produced for every library package and pushed
automatically together with the corresponding `.nupkg`.

## The manual equivalent

If you ever need to do it by hand:

```bash
# 1. Bump <Version> in Directory.Build.props and src/Shorokoo/Version.cs (keep in sync).

# 2. Test
dotnet test tests/Shorokoo.Tests/Shorokoo.Tests.csproj \
  --filter "Purpose=Coverage" -c Release

# 3. Pack everything
dotnet pack Shorokoo.sln -c Release -o artifacts/nupkgs

# 4. Push in dependency order (CodeGen and OnnxRuntime before the packages
#    that depend on them; --skip-duplicate makes re-runs safe)
for p in Shorokoo Shorokoo.Modules Shorokoo.CodeGen Shorokoo.OnnxRuntime \
         Shorokoo.LinuxCPU Shorokoo.LinuxGPU Shorokoo.WinCPU Shorokoo.WinGPU; do
  dotnet nuget push "artifacts/nupkgs/$p.<version>.nupkg" \
    --api-key "$NUGET_API_KEY" \
    --source https://api.nuget.org/v3/index.json --skip-duplicate
done

# 5. Tag and release
git tag v<version> && git push origin v<version>
```

## After a release

- Commit the version bump the app stamped into `Directory.Build.props` /
  `src/Shorokoo/Version.cs`.
- Check the package pages on nuget.org (readme, license, dependency lists render
  correctly). New packages can take a few minutes to index.
- Create a GitHub release for the tag.
