package com.bizlijakaria.tumerdtection;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence,Classified,confidancetxt;
    ImageView imageView;
    Button picture, upload,verify;
    int imageSize = 224;
    int[] intValues = new int[imageSize * imageSize];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        Classified=findViewById(R.id.classified);
        confidancetxt=findViewById(R.id.confidencesText);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        upload = findViewById(R.id.button2);
        verify=findViewById(R.id.button3);




        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        upload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Open gallery to select an image
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 2);

                new CountDownTimer(2000, 1000) {
                    public void onTick(long millisUntilFinished) {
                        // Do nothing here, as we want a 2-second timer
                    }

                    public void onFinish() {
                        verify.setVisibility(View.VISIBLE);
                    }
                }.start();
            }
        });


        verify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                verify.setVisibility(View.GONE);
                result.setVisibility(View.VISIBLE);
                confidancetxt.setVisibility(View.VISIBLE);
                confidence.setVisibility(View.VISIBLE);
                Classified.setVisibility(View.VISIBLE);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            // Capture image from camera
            Bitmap image = (Bitmap) data.getExtras().get("data");
            processAndClassifyImage(image);
        } else if (requestCode == 2 && resultCode == RESULT_OK && data != null) {
            // Select image from gallery
            Uri selectedImage = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                processAndClassifyImage(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void processAndClassifyImage(Bitmap image) {
        int dimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
        imageView.setImageBitmap(image);

        image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
        classifyImage(image);
    }

    public void classifyImage(Bitmap image){
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4); // Adjust the number of threads according to your device
            Interpreter model = new Interpreter(loadModelFile(), options);

            // Log message to indicate successful model loading
            Log.d("Model", "Model loaded successfully");

            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * imageSize * imageSize * 3 * 4); // 4 bytes per float
            inputBuffer.order(ByteOrder.nativeOrder());
            preprocessImage(image, inputBuffer);

            float[][] output = new float[1][2]; // Adjust the size according to your model output
            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(4 * 2); // 4 bytes per float
            outputBuffer.order(ByteOrder.nativeOrder());

            model.run(inputBuffer, outputBuffer);
            outputBuffer.rewind();
            outputBuffer.asFloatBuffer().get(output[0]);

            int maxIndex = argmax(output[0]);
            float confidenceValue = output[0][maxIndex];
            String[] classes = {"Yes","No"};
            String predictedClass = classes[maxIndex];

            result.setText(predictedClass);
            confidence.setText(String.format("%.1f%%", confidenceValue * 100));

            model.close();
        } catch (IOException e) {
            // Handle the exception
            Log.e("Model", "Error loading model: " + e.getMessage());
        }
    }


    private void preprocessImage(Bitmap bitmap, ByteBuffer buffer) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < imageSize; ++i) {
            for (int j = 0; j < imageSize; ++j) {
                final int val = intValues[pixel++];
                buffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                buffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                buffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
    }

    private int argmax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private ByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
