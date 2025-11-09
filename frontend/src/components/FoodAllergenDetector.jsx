
import React, { useState, useRef } from "react";
import { Upload, AlertCircle, CheckCircle, Camera, Loader2 } from "lucide-react";
import allergenData from "../allergens.json";

export default function FoodAllergenDetector() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [selectedAllergen, setSelectedAllergen] = useState("");
  const [allergenResult, setAllergenResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // ref for the hidden file input so we can clear it on reset
  const fileInputRef = useRef(null);

  const commonAllergens = [
    "dairy", "eggs", "fish", "shellfish",
    "tree nuts", "peanuts", "wheat",
    "gluten", "soy", "sesame"
  ];

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);

      // clear previous results
      setPrediction(null);
      setAllergenResult(null);
      setSelectedAllergen("");
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setPrediction(null);
    setAllergenResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Prediction failed");

      const result = await response.json();

      setPrediction({
        name: result.prediction.rawName.replace(/_/g, " "),
        confidence: (result.prediction.confidence * 100).toFixed(1),
        rawName: result.prediction.rawName.toLowerCase(),
      });

    } catch (error) {
      alert("Prediction failed. Check backend logs.");
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  function checkAllergen() {
    if (!prediction || !selectedAllergen) return;

    const foodKey = prediction.rawName.toLowerCase();  // "pizza"
    const foodData = allergenData[foodKey]?.allergens;

    if (!Array.isArray(foodData)) {
      setAllergenResult({ noData: true });
      return;
    }

    const contains = foodData.includes(selectedAllergen.toLowerCase());

    setAllergenResult({
      allergen: selectedAllergen,
      contains,
      allAllergens: foodData,
    });
  }

  const resetDetector = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setPrediction(null);
    setSelectedAllergen("");
    setAllergenResult(null);
    setIsAnalyzing(false);

    // Clear the file input value so the same file can be re-selected if needed
    if (fileInputRef.current) {
      fileInputRef.current.value = null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Camera className="w-10 h-10 text-emerald-600" />
            <h1 className="text-4xl font-bold text-slate-800">Food Allergen Detector</h1>
          </div>
          <p className="text-slate-600 text-lg">Upload a food image to identify it and check for allergens</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Left Column - Image Upload */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-slate-800 mb-4">Upload Image</h2>
            
            <label className="block cursor-pointer">
              <div className={`border-3 border-dashed rounded-xl transition-all ${imagePreview ? 'border-emerald-500 bg-emerald-50' : 'border-slate-300 bg-slate-50 hover:border-emerald-400 hover:bg-emerald-50'}`}>
                {imagePreview ? (
                  <div className="relative">
                    <img 
                      src={imagePreview} 
                      alt="Preview" 
                      className="w-full h-80 object-cover rounded-lg"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-10 transition-all rounded-lg flex items-center justify-center">
                      <span className="text-white opacity-0 hover:opacity-100 font-medium">
                        Click to change image
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="h-80 flex flex-col items-center justify-center">
                    <Upload className="w-16 h-16 text-slate-400 mb-3" />
                    <p className="text-slate-600 font-medium mb-1">Click to upload image</p>
                    <p className="text-slate-400 text-sm">PNG, JPG up to 10MB</p>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </label>

            <button
              onClick={analyzeImage}
              disabled={!selectedImage || isAnalyzing}
              className="w-full mt-4 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-all flex items-center justify-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                'Analyze Food'
              )}
            </button>

            {/* New Image / Reset button - visible only when an image is present */}
            {selectedImage && (
              <button
                onClick={resetDetector}
                className="w-full mt-2 bg-slate-200 hover:bg-slate-300 text-slate-800 font-semibold py-3 rounded-xl transition-all"
              >
                New Image
              </button>
            )}
          </div>

          {/* Right Column - Results */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-slate-800 mb-4">Detection Results</h2>
            
            {!prediction ? (
              <div className="h-80 flex items-center justify-center text-slate-400">
                <div className="text-center">
                  <Camera className="w-16 h-16 mx-auto mb-3 opacity-50" />
                  <p>Upload and analyze an image to see results</p>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Prediction Result */}
                <div className="bg-emerald-50 border-2 border-emerald-200 rounded-xl p-4">
                  <div className="flex items-start gap-3">
                    <CheckCircle className="w-6 h-6 text-emerald-600 mt-1 flex-shrink-0" />
                    <div className="flex-1">
                      <h3 className="font-semibold text-slate-800 text-lg mb-1">
                        Detected: {prediction.name}
                      </h3>
                      <p className="text-slate-600 text-sm">
                        Confidence: {prediction.confidence}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* Allergen Check */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Check for specific allergen
                  </label>
                  <div className="flex gap-2">
                    <select
                      value={selectedAllergen}
                      onChange={(e) => {
                        setSelectedAllergen(e.target.value);
                        setAllergenResult(null);
                      }}
                      className="flex-1 px-4 py-2 border-2 border-slate-300 rounded-lg focus:border-emerald-500 focus:outline-none"
                    >
                      <option value="">Select allergen...</option>
                      {commonAllergens.map(allergen => (
                        <option key={allergen} value={allergen}>
                          {allergen.charAt(0).toUpperCase() + allergen.slice(1)}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={checkAllergen}
                      disabled={!selectedAllergen}
                      className="px-6 py-2 bg-slate-700 hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-all"
                    >
                      Check
                    </button>
                  </div>
                </div>

                {/* Allergen Result */}
                {allergenResult && (
                  <div>
                    {allergenResult.noData ? (
                      <div className="bg-slate-50 border-2 border-slate-200 rounded-xl p-4">
                        <div className="flex items-start gap-3">
                          <AlertCircle className="w-6 h-6 text-slate-400 mt-1" />
                          <div>
                            <p className="text-slate-600">
                              No allergen data available for this food item
                            </p>
                          </div>
                        </div>
                      </div>
                    ) : allergenResult.contains ? (
                      <div className="bg-red-50 border-2 border-red-200 rounded-xl p-4">
                        <div className="flex items-start gap-3">
                          <AlertCircle className="w-6 h-6 text-red-600 mt-1 flex-shrink-0" />
                          <div>
                            <h4 className="font-semibold text-red-800 mb-1">
                              Warning: Contains {allergenResult.allergen}
                            </h4>
                            <p className="text-red-700 text-sm">
                              This food contains allergens you should avoid
                            </p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="bg-green-50 border-2 border-green-200 rounded-xl p-4">
                        <div className="flex items-start gap-3">
                          <CheckCircle className="w-6 h-6 text-green-600 mt-1 flex-shrink-0" />
                          <div>
                            <h4 className="font-semibold text-green-800 mb-1">
                              Safe: No {allergenResult.allergen} detected
                            </h4>
                            <p className="text-green-700 text-sm">
                              This food does not contain {allergenResult.allergen}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    {allergenResult.allAllergens && allergenResult.allAllergens.length > 0 && (
                      <div className="mt-3 p-3 bg-slate-50 rounded-lg">
                        <p className="text-sm font-medium text-slate-700 mb-2">
                          Known allergens in {prediction.name}:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {allergenResult.allAllergens.map(allergen => (
                            <span
                              key={allergen}
                              className="px-3 py-1 bg-white border border-slate-300 rounded-full text-sm text-slate-700"
                            >
                              {allergen}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Info Footer */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-4">
          <div className="flex gap-3">
            <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> This tool uses AI to detect food items and check allergens. 
              Always verify ingredients and consult with healthcare professionals for severe allergies.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
