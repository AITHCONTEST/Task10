import { makeAutoObservable } from "mobx";
import ITextStore from "../model/textStore";


class TextStore implements ITextStore{   

    private static instance: TextStore;

    public input: string;
    public output: string;

    private constructor() {
        this.input = "";
        this.output = "";
        this.loadFromLocalStorage();
        
        makeAutoObservable(this);
    }

    public static getInstance(): TextStore {
        if (!TextStore.instance) {
            TextStore.instance = new TextStore();
        }
        return TextStore.instance;
    }

    public getInput(): string {
        return this.input;
    }

    public getOutput(): string {
        return this.output;
    }


    public setInput(input: string): void {
        this.input = input;
        this.saveToLocalStorage();
    }

    public setOutput(output: string): void {
        this.output = output;
        this.saveToLocalStorage();
    }

    public setText(input: string, output: string): void {
        this.input = input;
        this.output = output;
        this.saveToLocalStorage();
    }

    private saveToLocalStorage(): void {
        localStorage.setItem('input', this.input);
        localStorage.setItem('output', this.output);
    }

    private loadFromLocalStorage(): boolean {
        const input = localStorage.getItem('input');
        if (input) {
            this.input = input;
        }else{
            this.input = "";
        }
        const output = localStorage.getItem('output');
        if (output) {
            this.output = output;
        }else{
            this.output = "";
        }
        return !!input && !!output;
    }


}

export default TextStore.getInstance();