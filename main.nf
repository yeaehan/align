nextflow.enable.dsl = 2

params.input = null
params.output = "results"
params.bucket = null
params.ref = "ch00"
params.mode = "2d"
params.no_gpu = false
params.disable_nonrigid = false
params.cpus = 4

if (!params.input) {
    error "Missing input folder. Use: --input /path/to/images"
}

process ALIGN_IMAGES {
    tag "${sample} (${params.mode})"

    cpus params.cpus

    publishDir "${params.output}/${sample}", mode: "copy", overwrite: true

    input:
    tuple val(sample), path(sample_files)

    output:
    tuple val(sample), path("aligned")

    script:
    def modeFlag = params.mode.toString().toLowerCase() == "3d" ? "--3d" : ""
    def gpuFlag = params.no_gpu ? "--no_gpu" : ""
    def nonrigidFlag = params.disable_nonrigid ? "--disable_nonrigid" : ""

    """
    export OMP_NUM_THREADS=${task.cpus}
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    align-pipeline \
        --input_folder . \
        --output_folder aligned \
        --ref "${params.ref}" \
        --n_workers ${task.cpus} \
        ${modeFlag} \
        ${gpuFlag} \
        ${nonrigidFlag}
    """
}

process UPLOAD_RESULTS {
    tag "upload: ${sample}"
    label "bucket_upload"

    publishDir "${params.output}/upload_receipts", mode: "copy", overwrite: true

    cpus 1
    memory "4 GB"
    time "6h"

    input:
    tuple val(sample), path(aligned_dir)

    output:
    path "${sample}.upload.complete"

    script:
    def bucket = params.bucket.toString().replaceAll('/+$', '')

    """
    gcloud storage rsync \
        "${aligned_dir}" \
        "${bucket}/${sample}" \
        --recursive

    printf 'sample=%s\\nbucket=%s\\ncompleted_utc=%s\\n' \
        "${sample}" \
        "${bucket}/${sample}" \
        "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${sample}.upload.complete"
    """
}

workflow {
    image_files = Channel
        .fromPath(
            "${params.input}/*.{tif,tiff,TIF,TIFF,lif,LIF}",
            checkIfExists: true
        )

    grouped_samples = image_files
        .map { file ->
            def matcher = file.name =~ /[A-Za-z]*\d+/
            def sample = matcher.find()
                ? matcher.group(0)
                : file.baseName.tokenize("_")[0]
            tuple(sample, file)
        }
        .groupTuple()
        .map { sample, files -> tuple(sample, files.sort { it.name }) }

    grouped_samples
        .map { sample, files ->
            "${sample},${files.collect { it.toString() }.join('|')}"
        }
        .collectFile(
            name: "generated_manifest.csv",
            newLine: true,
            seed: "sample,files\n",
            storeDir: params.output
        )

    aligned_samples = ALIGN_IMAGES(grouped_samples)

    if (params.bucket) {
        UPLOAD_RESULTS(aligned_samples)
    }
}
