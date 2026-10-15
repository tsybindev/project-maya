# gfx1030 (RDNA2 / RX 6800 XT): исследование поддержки в Maya

Дата: 2026-10-15. Машина: RX 6800 XT (gfx1030, 16 GB), ROCm 7.2.4 (Arch, `/opt/rocm`),
62 GB RAM, сеть есть, `sudo` в сессии нет. Maya v1.1.11.

## Вывод

Поддержка gfx1030 **реальна, это не «несобираемый» бэкенд**: CMake (`cmake/hip_backend.cmake`)
уже разрешает gfx1030 (в списке `_strata_hip_unvalidated`), рантайм-проверка
(`src/core/device.cu`) пропускает любую скомпилированную архитектуру, а почти весь движок
на gfx1030 падает на portable-пути автоматически. Режет только белый список в `maya.py`
(`HIP_ARCHS`) + одно жёсткое препятствие компиляции (rocWMMA, см. ниже).

## Что проверено экспериментально

| Факт | Как | Результат |
|---|---|---|
| Карта видна HIP | `hipcc`-проба `hipGetDeviceProperties` | `arch=gfx1030, warpSize=32` (wave32 ✓), LDS 64 KiB/блок ✓, 36 CU, 17.16 GB |
| `v_dot4_i32_i8` (`sdot4`) + `v_perm_b32` | проба `__builtin_amdgcn_sdot4` + `__builtin_amdgcn_perm` с `--offload-arch=gfx1030` | компилируется ✓ (декод пойдет через готовую ветку `intrinsics.hpp:26`) |
| rocWMMA под gfx1030 | проба `#include <rocwmma/rocwmma.hpp>` + `mma_sync` с `--offload-arch=gfx1030` | **FAIL: `config.hpp:88 static_assert "Unsupported architecture"`** — rocWMMA знает только CDNA (gfx908/90a/942/950) и gfx11/gfx12. Это единственное жёсткое препятствие |
| Заголовков rocWMMA в системном ROCm нет | `ls /opt/rocm/include` | нет; `pacman -Ss rocwmma` → пакет `rocwmma 7.2.4-1` есть в репозиториях, нужна установка (`sudo pacman -S rocwmma`) — в этой сессии `sudo` недоступен |
| rocBLAS под gfx1030 | `ls /opt/rocm/lib/rocblas/library` | 88 файлов `*gfx1030*` ✓ — hipBLAS GEMM (в т.ч. FP16) работать будет |
| Биткод устройства | `/opt/rocm/amdgcn/bitcode/` | `oclc_isa_version_1030.bc` есть ✓ |
| hipBLASLt-таблиц под gfx1030 нет | `ls tools/hip` | только gfx1100/1151/1201 → движок возьмёт обычный hipBLAS (штатный фолбэк, править не надо) |
| `wall_clock64()` | используется в `glm_fast_path.cu`, `verify_kernels.cu` | это обёртка HIP (`hip/amd_detail/amd_device_functions.h`), доступна на всех AMD ✓ |

## Почему движок в основном готов (анализ кода)

- Декод: `rdna3_expert_device()` (`glm_fast.cu:4689`) сравнивает строку строго с
  `gfx1100`/`gfx1151` → на gfx1030 `false` → используются обычные expert-kernels.
  `q6_idot` (`glm_fast.cu:1761`) на gfx1030 идёт в `#else` → `__dp4a` → `sdot4` ✓.
- Префилл fused-эксперты (`moe_fused_rdna4.cu`): `device_info()` (`:640`) включает путь
  только при `gcnArchName == gfx1151`; `native_kernel` для gfx1030 компилируется в ветку
  `__builtin_trap()` (не вызывается) ✓. `__builtin_amdgcn_perm` на gfx1030 есть ✓.
- Префилл attention (`glm_prefill.cu:157` + `glm_batch.cu:1486`): `mla_wmma2_arch()`
  вернёт `3` («anything else») → `supported()==false`, `default()==false` → выбор падает на
  `F16Q` (rocWMMA) или `WMMA2`. Оба rocWMMA/WMMA-пути на gfx1030 невозможны →
  **нужно форсировать `MlaAttnPath::F32`** (обычный `mla_attn_kernel`, ~48 KB shared,
  архитектурно-независимый). FP16-GEMM через hipBLAS при этом остаются (rocBLAS знает gfx1030).
- `mla2_wmma` (`glm_batch.cu:1354`): `#else __builtin_trap()` — компилируется ✓, не вызывается ✓.
- `mv_rdna_kernel` (STRATA_GLM_MV_RDNA): включается только для `gfx11*`/`gfx115*`
  (`glm_fast.cu:4041`, префикс `"gfx11"`, `"gfx1030"` не подходит) ✓.
- `native_qsa_score.cu:144`: HIP-ветка — скалярный F32 fallback ✓.
- ggml-MMQ: кодировка `ggml_cuda_parse_amd_id` (`ggml_cuda_host.cu:34`) универсальна
  (gfx1030 → cc `0x01001030`); llama.cpp HIP официально умеет RDNA2 → должен скомпилироваться,
  **проверить реальной сборкой**.
- MTP-decode (`glm_mtp.cu:89`): только `gfx115*` → на gfx1030 выключен штатно ✓
  (декод без драфтинга, как на старых NVIDIA).
- `setup.py`: имя карты неизвестно (`AMD_NAMES` без gfx1030) — отсюда «AMD Radeon (gfx1030)»
  в логе; KFD даёт arch напрямую, для Linux правится одной строкой. PCI-ID/Navi2x для
  Windows-пути: Navi21 `73bf/73af` → gfx1030, Navi22 `73df` → gfx1031, Navi23 `73ff` → gfx1032.

## План правок (статус: пункты 1–6 выполнены 2026-10-15, сборка ещё не запускалась)

1. ✅ `maya.py`: `HIP_ARCHS += ("gfx1030", "gfx1031", "gfx1032")` + тексты (шапка, step 1,
   fail-сообщение, `--backend help`).
2. ✅ `setup.py`: `AMD_NAMES` + `AMD_DEVICE_ARCHS` + `AMD_NAME_ARCHS` для Navi2x
   (Navi21 `73bf/73af` → gfx1030, Navi22 `73df` → gfx1031, Navi23 `73ff` → gfx1032;
   `RX 68x0/69x0` → gfx1030, `RX 67x0` → gfx1031, `RX 66x0` → gfx1032).
3. ✅ `src/kernels/cuda/glm_batch.cu`:
   - `#include <rocwmma/rocwmma.hpp>` ограждён: host-проход и gfx11/12 device-проходы
     включают, gfx103x device-проход — нет; введён маркер `STRATA_HIP_ROCWMMA`.
   - `mla_attn_wmma_kernel` — только при `STRATA_HIP_ROCWMMA`.
   - `mla_attn_f16q`: при `STRATA_HIP_ROCWMMA` — как раньше; без него — host-стаб
     (лог + возврат, выход не трогает).
   - Добавлен `bool mla_f16q_supported()` (объявлен в `include/strata/kernels/glm_batch.hpp`);
     `mla_wmma2_supported()/default()` тоже возвращают false без rocWMMA-кода.
   - `mla2_wmma`: исходный `#else __builtin_trap()` оставлен (компилируется на gfx103x,
     рантайм туда не ходит: `mla_wmma2_supported()==false`).
4. ✅ `src/core/glm_prefill.cu` (`mla_attn_path`): если оба `*_supported()==false` → `F32`
   (со stderr-заметкой при явном `STRATA_GLM_PREFILL_ATTN=f16q/wmma2`). Call site
   (`:1873`) безопасен: при F32 вызывается только `gb::mla_attn`.
5. ✅ `tests/hip/glm_prefill_attention.cpp`: WMMA-секция скипается через
   `mla_f16q_supported()` (wmma2-секция уже скипалась через `mla_wmma2_supported()`).
6. ✅ `tools/test_maya_hip.py`: обе захардкоженные строки `-DCMAKE_HIP_ARCHITECTURES=…`
   заменены на `";".join(maya.HIP_ARCHS)`; новый тест
   `test_rdna2_card_is_usable_without_tuning_table` (gfx1030 принимается, конфиг HIP,
   без hipBLASLt-таблицы и без PREFILL_SUB-оверрайда). `tools/test_hip_archs.py` не тронут.
7. ✅ `docs/AMD_MAYA.md`: gfx103x в заголовке/списке + секция «RDNA2 (gfx103x, experimental)».
8. Валидация:
   - ✅ 4 targeted-теста (`rdna2`, `selects_larger`, `rejects_unsupported`, `hip_build_mmq`) — OK.
   - ✅ `tools/test_hip_archs.py` — 5/5 OK (не трогали).
   - ⚠️ Полный `tools/test_maya_hip` падает в `test_cli_backend_never_benchmarks_cuda_config_as_hip`
     («Maya is not set up here yet») — **предсуществующее**, на чистом main через `git stash`
     падает так же (логи идентичны кроме tmp-имён).
   - ✅ **Реальная сборка под gfx1030 на RX 6800 XT (ROCm 7.2.4, системный `rocwmma`):**
     `cmake -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON
     -DSTRATA_PORTABLE=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030` → configure OK (штатный
     warning «not validated»), `strata_kernels` OK (только pre-existing warnings),
     `hip_glm_prefill_attention` OK + **ран на живой карте: F32 max abs err 3.9e-08
     (tolerance 5e-5), обе WMMA-секции корректно скипнуты, PASS**.
     Полный `strata` (29.9 MB) собран без ошибок (1934 warning'а — все nodiscard/unused,
     pre-existing), `strata-device` на живой карте:
     `gfx1030 wave32 / gfx1030, 36 CU, 15.98 GB` ✓.
     По ходу чинены две ошибки guard'ов: опечатка `STATA_USE_HIP` в ручной команде и
     guard, проглотивший `mla_attn_wmma2_kernel` (ядро на чистых builtin'ах — компилируется
     везде, guard нужен только rocWMMA-ядру `mla_attn_wmma_kernel`; итог: `#endif // STRATA_HIP_ROCWMMA`
     сразу после него + `#if defined(STRATA_USE_HIP)` перед `using mla2_h16`).
   - ✅ **Починка вдогонку (FATAL_ERROR при реальной установке):** `HIP_ARCHS` в `maya.py`
     содержал `gfx1032`, а `_strata_hip_unvalidated` в `cmake/hip_backend.cmake` — нет →
     setup под все 6 archs падал на configure. Добавлены `gfx1032 gfx1035 gfx1036`
     (Navi23/Navi24 + GFX1035 APU, тот же wave32/RDNA2 класс, portable kernels).
   - ⬜ `./maya.sh --backend hip --gpu 0 --check` на реальной машине (теперь setup пропустит карту).
   - ✅ **Полный `ctest` в build-1030: 49/51 passed (96%), 2 скипа — оба штатные:**
     `hip_prefill_hipblaslt_gemm` (Skipped — нет таблицы под gfx1030, ожидаемо),
     `hip_prefill_fused_parity` (Skipped — fused-путь только для gfx1151, ожидаемо).
     2 FAIL — оба окружение, не код: `ple_parity` (нет файла весов
     `Qwen3.8-Flash-Next-Q2_0-*.gguf` — нужны скачанные веса), `platform_memory_test`
     (`mlock failed (raise ulimit -l)` — ограничение контейнера/сессии).
     Все GPU-тесты зелёные, включая `hip_prefill_mmq_parity` (R1 закрыт по-настоящему),
     `hip_device_selftest`, `qsa_parity`, `kv_*_parity`.
   - ✅ **`./maya.sh --backend hip --gpu 0 --check` на реальной машине — ПРОХОДИТ:**
     `GPU 0 (AMD Radeon RX 6800 / 6900 series (gfx1030), 16 GB, gfx1030) - can be used`,
     `using GPU 0 ... experimental HIP, one GPU, text only`, ROCm /opt/rocm, RAM 63/53 GB.
   - ⬜ Полный прогон модели (нужны ~96.5 GB весов / `--gguf-dir`).

## Открытые риски (для следующей сессии)

- R1: ✅ **Закрыт**: ggml-MMQ пиннутого llama.cpp (`3cf03257`) под gfx1030 скомпилировался
  в составе полного `strata` без ошибок.
- R4: ✅ **Закрыт**: пользователь установил `rocwmma` в систему.
- R2: скорость F32-attention + plain hipBLAS на 16 GB: 321B-модель погонится, но медленно;
  ожидания ставить низкие (это тест запуска, не бенчмарк).
- R3: 16 GB VRAM + Maya-S 96.5 GB: почти всё поедет из RAM/SSD — нужен быстрый NVMe и
  свободная RAM (тут 55 GB avail — S24/M-компактные кванты предпочтительнее).
- R5: ✅ **Закрыт**: полный `ctest` в build-1030 — 49/51, все GPU-тесты зелёные,
  2 FAIL только по окружению (веса/ulimit), 2 скипа штатные.
- Что осталось: полный прогон модели (`--download-model` ~96.5 GB или `--gguf-dir`
  с готовыми весами) — за рамками этой сессии по объёму скачивания.
