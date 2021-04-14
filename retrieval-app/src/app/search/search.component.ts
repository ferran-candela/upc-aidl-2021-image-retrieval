import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from "@angular/forms";
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.scss']
})
export class SearchComponent implements OnInit {

  SERVER_URL = "/api";
  SEARCH_URL = this.SERVER_URL + "/search"
  MODELS_URL = this.SERVER_URL + "/models"
  IMAGES_URL = this.SERVER_URL + "/images"

  uploadForm: FormGroup;
  imageURL: string;
  ranking: number[];
  models: string[]

  constructor(
    private formBuilder: FormBuilder,
    private httpClient: HttpClient
  ) {
    this.imageURL = '';
    this.ranking = [];
    this.models = [];
    this.uploadForm = this.formBuilder.group({
      fileName: 'Select image',
      searchImage: [''],
      model: 'resnet50_custom',
      topK: '15'
    });
  }

  ngOnInit(): void {
    this.loadModels();
  }

  onFileSelect(event: any) {

    if (event.target.files.length > 0) {
      const file = event.target.files[0];

      this.uploadForm.patchValue({
        "fileName": file.name
      });

      let searchImage = this.uploadForm.get('searchImage');

      if(searchImage != undefined) {
        searchImage.setValue(file);
      }

      // File Preview
      const reader = new FileReader();
      reader.onload = () => {
        this.imageURL = reader.result as string;
      }

      reader.readAsDataURL(file)
    }
  }

  loadModels() {
    this.httpClient.get<any>(this.MODELS_URL).subscribe(
      (res) => {
        this.models = res.models !== undefined ? res.models : [];
      },
      (err) => console.log(err)
    );
  }

  onSubmit() {
    const formData = new FormData();

    let searchImage = this.uploadForm.get('searchImage');
    let model = this.uploadForm.get('model');
    let topK = this.uploadForm.get('topK');


    if(searchImage != undefined && model != undefined && topK != undefined) {  
      formData.append('image', searchImage.value);
      formData.append('model', model.value);
      formData.append('topK', topK.value);

      this.httpClient.post<any>(this.SEARCH_URL, formData).subscribe(
        (res) => {
          if(res.success === true) {
            this.ranking = res.ranking;
          } else {
            this.ranking = []
          }          
        },
        (err) => console.log(err)
      );
    }


  }
}
