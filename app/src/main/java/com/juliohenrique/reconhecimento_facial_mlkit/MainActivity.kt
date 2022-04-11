package com.juliohenrique.reconhecimento_facial_mlkit

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.util.Pair
import android.view.View
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import com.google.firebase.crashlytics.buildtools.reloc.com.google.common.reflect.TypeToken
import com.google.gson.Gson
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.juliohenrique.reconhecimento_facial_mlkit.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.ReadOnlyBufferException
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.experimental.inv

class MainActivity : AppCompatActivity() {

    private var TAG = "MainActivity"
    private val binding by lazy { ActivityMainBinding.inflate(layoutInflater) }
    private val MY_CAMERA_REQUEST_CODE = 100
    private var imageCapture: ImageCapture? = null
    private var camFace = CameraSelector.LENS_FACING_FRONT
    private var flipX = false
    private var start = true
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var cameraSelector: CameraSelector
    private lateinit var preview: Preview
    private lateinit var previewView: PreviewView
    private lateinit var detector: FaceDetector

    private var distance = 1.0f
    private lateinit var embeedings: Array<FloatArray>
    private var inputSize = 112
    private var isModelQuantized = false
    private var developerMode = false
    private var IMAGE_MEAN = 128.0f
    private var IMAGE_STD = 128.0f
    private var OUTPUT_SIZE = 192
    private val SELECT_PICTURE = 1
    private lateinit var intValues: IntArray
    private lateinit var tfLite: Interpreter

    private var modelFile = "mobile_face_net.tflite"

    private var registered = HashMap<String, SimilarityClassifier.Recognition>() // Salva rostos

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        registered = readFromSP()!!
        setContentView(binding.root)

        val sharedPref = getSharedPreferences("Distance", MODE_PRIVATE)
        distance = sharedPref.getFloat("distance", 1.00f)

        binding.imageView.visibility = View.INVISIBLE
        binding.btnReconhecer.text = "ADD ROSTO"
        binding.btnAddrosto.visibility = View.INVISIBLE

        solicitaPermissao()
        configuraBotaoAddRosto()
        configuraBotaoReconhecimento()
        carregaModeloBitmap()
        configuraFaceDetector()

//        Analisar para ativar a detecção do contorno do rosto.
//        val realTimeOpts = FaceDetectorOptions.Builder()
//            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
//            .build()
//        detector = FaceDetection.getClient(realTimeOpts)

        abreCamera();
        alternaCamera();
    }

    private fun configuraFaceDetector() {
        //Inicializa e configura o Face Detector
        val highAccuracyOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build()
        detector = FaceDetection.getClient(highAccuracyOpts)
    }

    private fun carregaModeloBitmap() {
        //Carrega modelo de arquivo do bitmap
        try {
            tfLite = Interpreter(loadModelFile(this@MainActivity, modelFile)!!)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun configuraBotaoReconhecimento() {
        binding.btnReconhecer.setOnClickListener {
            configuraComponentes()
        }
    }

    private fun configuraBotaoAddRosto() {
        binding.btnAddrosto.setOnClickListener {
            addRosto()
        }
    }

    private fun solicitaPermissao() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                arrayOf(Manifest.permission.CAMERA),
                MY_CAMERA_REQUEST_CODE
            )
        }
    }

    private fun developerMode() {
        if (developerMode) {
            developerMode = false
            toast("Developer Mode OFF")
        } else {
            developerMode = true
            toast("Developer Mode ON")
        }
    }

    private fun configuraComponentes() {
        if (binding.btnReconhecer.text == "Reconhecer") {
            start = true
            //textAbove_preview.setText("Recognized Face:")
            binding.btnReconhecer.text = "ADD ROSTO"
            binding.btnAddrosto.visibility = View.INVISIBLE
            binding.txvRecoName.visibility = View.VISIBLE
            //preview_info.setText("")
            //preview_info.setVisibility(View.INVISIBLE);
        } else {
            // textAbove_preview.setText("Face Preview: ")
            binding.btnReconhecer.text = "Reconhecer"
            binding.btnAddrosto.visibility = View.VISIBLE
            binding.txvRecoName.visibility = View.INVISIBLE
            binding.imageView.visibility = View.VISIBLE

            //preview_info.setText("1.Bring Face in view of Camera.\n\n2.Your Face preview will appear here.\n\n3.Click Add button to save face.")
        }


    }

    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity, MODEL_FILE: String): MappedByteBuffer? {
        val fileDescriptor = activity.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    //Carrega imagens do Shared para reconhecimento facial
    private fun readFromSP(): HashMap<String, SimilarityClassifier.Recognition>? {
        val sharedPreferences = getSharedPreferences("HashMap", MODE_PRIVATE)
        val defValue = Gson().toJson(HashMap<String, SimilarityClassifier.Recognition>())
        val json = sharedPreferences.getString("map", defValue)
        Log.i(TAG, "readFromSP: Output json ${json.toString()}")
        val token: TypeToken<HashMap<String?, SimilarityClassifier.Recognition?>?> =
            object : TypeToken<HashMap<String?, SimilarityClassifier.Recognition?>?>() {}
        val retrievedMap: HashMap<String, SimilarityClassifier.Recognition> =
            Gson().fromJson<HashMap<String, SimilarityClassifier.Recognition>>(json, token.type)
        Log.i(TAG, "readFromSP: Output map ${retrievedMap.toString()}")

        for ((_, value) in retrievedMap) {
            val output = Array(1) {
                FloatArray(
                    OUTPUT_SIZE
                )
            }
            var arrayList = value.getExtra() as ArrayList<*>
            arrayList = arrayList[0] as ArrayList<*>
            for (counter in arrayList.indices) {
                output[0][counter] = (arrayList[counter] as Double).toFloat()
            }
            value.setExtra(output)
        }
        toast("Reconhecimentos carregados")
        return retrievedMap
    }

    private fun inserirNoSP(jsonMap: HashMap<String, SimilarityClassifier.Recognition>, mode: Int) {
        if (mode == 1) //mode: 0:save all, 1:clear all, 2:update all
            jsonMap.clear() else if (mode == 0) jsonMap.putAll(readFromSP()!!)
        val jsonString = Gson().toJson(jsonMap)

        val sharedPreferences = getSharedPreferences("HashMap", MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        editor.putString("map", jsonString)
        Log.i(TAG, "inserirNoSP: Input josn ${jsonString.toString()}")
        editor.apply()
        toast("Salvo no Shared")
    }

    @SuppressLint("UnsafeOptInUsageError", "UnsafeExperimentalUsageError")
    private fun vinculaCamera(cameraProvider: ProcessCameraProvider) {
        val preview = Preview.Builder()
            .build()
        //Define a câmera frontal/traseira
        cameraSelector = CameraSelector.Builder()
            .requireLensFacing(camFace)
            .build()

        preview.setSurfaceProvider(previewView.surfaceProvider)
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        val executor: Executor = Executors.newSingleThreadExecutor()
        imageAnalysis.setAnalyzer(executor, { imageProxy ->
            try {
                Thread.sleep(10) //Camera preview atualiza sempre a cada 10 millisec
            } catch (e: InterruptedException) {
                e.printStackTrace()
            }
            // A imagem deve ser tratada aqui
            lateinit var image: InputImage
            val mediaImage: Image? = imageProxy.image

            if (mediaImage != null) {
                //Rotação da imagem
                image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                Log.i(TAG, "vinculaCamera: ${imageProxy.imageInfo.rotationDegrees}")
            }

            //Processa a imagem
            val result = detector.process(image)
                .addOnSuccessListener { faces ->
                    if (faces.size != 0) {
                        // Obtém o rosto
                        val face = faces[0]
                        Log.i("Main", "onSuccess: $face")

                        var frameBmp: Bitmap? = toBitmap(mediaImage)

                        val rot = imageProxy.imageInfo.rotationDegrees

                        //Adjust orientation of Face
                        val frameBmp1: Bitmap? = rotateBitmap(
                            frameBmp,
                            rot,
                            flipX = false,
                            flipY = false
                        )

                        //Pega o rosto no limite como se fosse uma caixa
                        val boundingBox = RectF(face.boundingBox)

                        //Recortar a caixa delimitadora de todo o Bitmap (imagem)
                        var croppedFace: Bitmap? = getCropBitmapByCPU(frameBmp1, boundingBox)

                        if (flipX) croppedFace = rotateBitmap(croppedFace, 0, flipX, false)
                        //Escala o rosto para 112*112 que é a entrada necessária para o modelo
                        val scaled: Bitmap? = getResizedBitmap(croppedFace, 112, 112)

                        if (start)
                            reconhecerImagem(scaled!!) //Envie bitmap dimensionado para criar integhração de face.
                        Log.i(TAG, "vinculaCamera boundingBox: $boundingBox")

                    } else {
                        if (registered.isEmpty())
                            binding.txvRecoName.text = "Adicione um rosto"
                        else
                            binding.txvRecoName.text = "Nenhum rosto detectado."

                    }
                }
                .addOnFailureListener {
                    Log.i(TAG, "vinculaCamera: onFailed")
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        })

        cameraProvider.bindToLifecycle(
            (this as LifecycleOwner)!!,
            cameraSelector,
            imageAnalysis,
            preview
        )
    }

    private fun findNearest(emb: FloatArray): List<Pair<String, Float>?>? {
        val neighbour_list: MutableList<Pair<String, Float>?> = ArrayList()
        var ret: Pair<String, Float>? = null //to get closest match
        var prev_ret: Pair<String, Float>? = null //to get second closest match
        for ((name, value) in registered) {
            val knownEmb = (value.getExtra() as Array<FloatArray>)[0]
            var distance = 0f
            for (i in emb.indices) {
                val diff = emb[i] - knownEmb[i]
                distance += diff * diff
            }
            distance = Math.sqrt(distance.toDouble()).toFloat()
            if (ret == null || distance < ret.second) {
                prev_ret = ret
                ret = Pair(name, distance)
            }
        }
        if (prev_ret == null) prev_ret = ret
        neighbour_list.add(ret)
        neighbour_list.add(prev_ret)
        return neighbour_list
    }

    fun reconhecerImagem(bitmap: Bitmap) {
        //Seta Imagem preview
        binding.imageView.setImageBitmap(bitmap)

        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        imgData.order(ByteOrder.nativeOrder())
        intValues = IntArray(inputSize * inputSize)

        // Pega os pixel do bitmap
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        imgData.rewind()
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue: Int = intValues.get(i * inputSize + j)
                if (isModelQuantized) {
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else { // Float model
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }


        //imgData entrada para o modelo
        val inputArray = arrayOf<Any>(imgData)
        val outputMap: MutableMap<Int, Any> = HashMap()
        embeedings =
            Array(1) { FloatArray(OUTPUT_SIZE) } // Armazena o modelo
        outputMap[0] = embeedings
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap) //Run model
        var distanciaLocal = Float.MAX_VALUE
        val id = "0"
        val label = "?"

        //Compara o rosto com o rosto armazenado.
        if (registered.size > 0) {
            val nearest: List<Pair<String, Float>?> =
                findNearest(embeedings[0])!! //Encontra dois rostos correspondentes.
            if (nearest[0] != null) {
                val name = nearest[0]!!.first //Obtém o nome e a distância do rosto mais próximo.
                distanciaLocal = nearest[0]!!.second
                if (developerMode) {
                    if (distanciaLocal < distance) //Verifica se a distância entre o rosto mais próximo encontrado for maior que 1, então a saída deve ser UNKDOWN.
                        binding.txvRecoName.setText(
                            """
                        Nearest: $name
                        Dist: ${String.format("%.3f", distanciaLocal)}
                        2nd Nearest: ${nearest[1]!!.first}
                        Dist: ${String.format("%.3f", nearest[1]!!.second)}
                        """.trimIndent()
                        ) else binding.txvRecoName.setText(
                        """
                        Unknown
                        Dist: ${String.format("%.3f", distanciaLocal)}
                        Nearest: $name
                        Dist: ${String.format("%.3f", distanciaLocal)}
                        2nd Nearest: ${nearest[1]!!.first}
                        Dist: ${String.format("%.3f", nearest[1]!!.second)}
                        """.trimIndent()
                    )

                    Log.i(TAG, "recognizeImage:   $name  - distance:  $distanciaLocal")
                } else {
                    if (distanciaLocal < distance) {//If distance between Closest found face is more than 1.000 ,then output UNKNOWN face.
                        binding.txvRecoName.text = name
                    } else {
                        binding.txvRecoName.text = "Unknown"
                    }

                    Log.i(TAG, "recognizeImage:   $name  - distance:  $distanciaLocal")
                }
            }
        }
    }

    fun getResizedBitmap(bm: Bitmap?, newWidth: Int, newHeight: Int): Bitmap? {
        val width = bm!!.width
        val height = bm!!.height
        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height
        // CREATE A MATRIX FOR THE MANIPULATION
        val matrix = Matrix()
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight)

        // "RECREATE" THE NEW BITMAP
        val resizedBitmap = Bitmap.createBitmap(
            bm, 0, 0, width, height, matrix, false
        )
        bm.recycle()
        return resizedBitmap
    }


    private fun getCropBitmapByCPU(source: Bitmap?, cropRectF: RectF): Bitmap? {
        val resultBitmap = Bitmap.createBitmap(
            cropRectF.width().toInt(),
            cropRectF.height().toInt(), Bitmap.Config.ARGB_8888
        )
        val cavas = Canvas(resultBitmap)

        // draw background
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        paint.color = Color.WHITE
        cavas.drawRect(
            RectF(0f, 0f, cropRectF.width(), cropRectF.height()),
            paint
        )
        val matrix = Matrix()
        matrix.postTranslate(-cropRectF.left, -cropRectF.top)
        cavas.drawBitmap(source!!, matrix, paint)
        if (source != null && !source.isRecycled) {
            source.recycle()
        }
        return resultBitmap
    }

    private fun toBitmap(image: Image?): Bitmap? {

        val nv21: ByteArray? = YUV_420_888toNV21(image)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image!!.width, image!!.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 75, out)
        val imageBytes = out.toByteArray()

        // bytes
        Log.i(TAG, "toBitmap: ${Arrays.toString(imageBytes)}")

        // formato
        Log.i(TAG, "toBitmap: ${image.format}")

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    // Esse método realiza a conversão toBitmap excepcionalmente
    private fun YUV_420_888toNV21(image: Image?): ByteArray? {
        val width = image!!.width
        val height = image!!.height
        val ySize = width * height
        val uvSize = width * height / 4
        val nv21 = ByteArray(ySize + uvSize * 2)
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V
        var rowStride = image.planes[0].rowStride
        assert(image.planes[0].pixelStride == 1)
        var pos = 0
        if (rowStride == width) { // likely
            yBuffer[nv21, 0, ySize]
            pos += ySize
        } else {
            var yBufferPos = -rowStride.toLong() // not an actual position
            while (pos < ySize) {
                yBufferPos += rowStride.toLong()
                yBuffer.position(yBufferPos.toInt())
                yBuffer[nv21, pos, width]
                pos += width
            }
        }
        rowStride = image.planes[2].rowStride
        val pixelStride = image!!.planes[2].pixelStride
        assert(rowStride == image!!.planes[1].rowStride)
        assert(pixelStride == image!!.planes[1].pixelStride)
        if (pixelStride == 2 && rowStride == width && uBuffer[0] == vBuffer[1]) {
            val savePixel = vBuffer[1]
            try {
                vBuffer.put(1, savePixel.inv())
                if (uBuffer[0] == savePixel.inv()) {
                    vBuffer.put(1, savePixel)
                    vBuffer.position(0)
                    uBuffer.position(0)
                    vBuffer[nv21, ySize, 1]
                    uBuffer[nv21, ySize + 1, uBuffer.remaining()]
                    return nv21 // shortcut
                }
            } catch (ex: ReadOnlyBufferException) {

            }
            vBuffer.put(1, savePixel)
        }

        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                val vuPos = col * pixelStride + row * rowStride
                nv21[pos++] = vBuffer[vuPos]
                nv21[pos++] = uBuffer[vuPos]
            }
        }
        return nv21
    }

    private fun addRosto() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Digite seu nome:")

        // Configura input do Dialog
        val input = EditText(this)

        input.inputType = InputType.TYPE_CLASS_TEXT
        builder.setView(input)

        // Configura botão positivo
        builder.setPositiveButton("ADICIONA") { _, _ ->
            toast(input.text.toString())
            //Cria e inicializa um novo objeto.
            val result = SimilarityClassifier.Recognition(
                "0", "", -1f
            )
            result.setExtra(embeedings)
            registered.put(input.text.toString(), result)
            inserirNoSP(registered, 0)
            start = true
        }
        builder.setNegativeButton("Cancela") { dialog, _ ->
            start = true
            dialog.cancel()
        }
        builder.show()
    }

    private fun abreCamera() {
        cameraProviderFuture = ProcessCameraProvider
            .getInstance(this)

        previewView = binding.cameraPreview

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                vinculaCamera(cameraProvider)
            } catch (e: Exception) {
                Log.i(TAG, "startCamera: startCamera fail:", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Alterna entre a câmera frontal e traseira
    private fun alternaCamera() {
        binding.alteraCamera.setOnClickListener {
            if (camFace == CameraSelector.LENS_FACING_FRONT) {
                camFace = CameraSelector.LENS_FACING_BACK
                flipX = true
            } else {
                camFace = CameraSelector.LENS_FACING_FRONT
                flipX = false
            }
            cameraProvider.unbindAll();
            abreCamera()
        }
    }

    private fun rotateBitmap(
        bitmap: Bitmap?, rotationDegrees: Int, flipX: Boolean, flipY: Boolean
    ): Bitmap? {
        val matrix = Matrix()

        // Gira a imagem de volta.
        matrix.postRotate(rotationDegrees.toFloat())

        // Espelha a imagem ao longo do eixo X ou Y.
        matrix.postScale(if (flipX) -1.0f else 1.0f, if (flipY) -1.0f else 1.0f)
        val rotatedBitmap =
            Bitmap.createBitmap(bitmap!!, 0, 0, bitmap.width, bitmap.height, matrix, true)

        // Recicla o bitmap antigo se ele foi alterado.
        if (rotatedBitmap != bitmap) {
            bitmap.recycle()
        }
        return rotatedBitmap
    }

    // Verifica o resultado da permissão solicitada
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String?>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == MY_CAMERA_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                toast("Permissão concedida")
            } else {
                toast("Permissão negada")
            }
        }
    }

    private fun toast(mensagem: String) {
        Toast.makeText(
            this,
            mensagem,
            Toast.LENGTH_SHORT
        ).show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                val selectedImageUri = data!!.data
                try {
                    val impphoto = InputImage.fromBitmap(getBitmapFromUri(selectedImageUri), 0)
                    detector.process(impphoto).addOnSuccessListener { faces ->
                        if (faces.size != 0) {
                            binding.btnReconhecer.text = "Recognize"
//                            add_face.setVisibility(View.VISIBLE)
//                            reco_name.setVisibility(View.INVISIBLE)
//                            face_preview.setVisibility(View.VISIBLE)
//                            preview_info.setText("1.Bring Face in view of Camera.\n\n2.Your Face preview will appear here.\n\n3.Click Add button to save face.")
                            val face = faces[0]
                            Log.i(TAG, "onActivityResult: $face")

                            //Para mostrar o bitmap na tela
                            var frame_bmp: Bitmap? = null
                            try {
                                frame_bmp = getBitmapFromUri(selectedImageUri)
                            } catch (e: IOException) {
                                e.printStackTrace()
                            }
                            val frame_bmp1: Bitmap = rotateBitmap(frame_bmp, 0, flipX, false)!!

                            //face_preview.setImageBitmap(frame_bmp1);
                            val boundingBox = RectF(face.boundingBox)
                            val cropped_face: Bitmap =
                                this.getCropBitmapByCPU(frame_bmp1, boundingBox)!!
                            val scaled = getResizedBitmap(cropped_face, 112, 112)
                            // face_preview.setImageBitmap(scaled);
                            reconhecerImagem(scaled!!)
                            addRosto()
                            Log.i(TAG, "onActivityResult: $boundingBox")
                            try {
                                Thread.sleep(100)
                            } catch (e: InterruptedException) {
                                e.printStackTrace()
                            }
                        }
                    }.addOnFailureListener {
                        start = true
                        toast("Falha ao adicionar imagem")
                    }
                    binding.imageView.setImageBitmap(getBitmapFromUri(selectedImageUri))
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun getBitmapFromUri(uri: Uri?): Bitmap? {
        val parcelFileDescriptor = contentResolver.openFileDescriptor(uri!!, "r")
        val fileDescriptor = parcelFileDescriptor!!.fileDescriptor
        val image = BitmapFactory.decodeFileDescriptor(fileDescriptor)
        parcelFileDescriptor.close()
        return image
    }
}