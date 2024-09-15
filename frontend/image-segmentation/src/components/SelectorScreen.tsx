import { Button } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import React, { useState, ChangeEvent } from "react";

const ScreenSelector = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
    }
  };

  return (
    <div className="container mx-auto p-6 min-h-screen flex flex-col items-center justify-center bg-gray-100">
      <div className="w-full max-w-md bg-white rounded-lg shadow-xl p-6 space-y-6">
        <h1 className="text-2xl font-semibold text-gray-800 text-center">
          Upload an Image
        </h1>
        <Button
          component="label"
          variant="contained"
          startIcon={<CloudUploadIcon />}
        >
          Upload files
          {/* Hidden input element AAAAAAAAAAAAAAAAAAAAAAAAAAAA */}
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            multiple
            style={{ display: "none" }}
          />
        </Button>
      </div>
      {selectedImage && (
        <div className="mt-8 w-full max-w-lg bg-white p-4 rounded-lg shadow-lg">
          <img
            src={selectedImage}
            alt="Uploaded"
            className="w-full h-auto object-contain rounded-lg shadow-md transition-transform duration-500 hover:scale-105"
          />
        </div>
      )}
    </div>
  );
};

export default ScreenSelector;
