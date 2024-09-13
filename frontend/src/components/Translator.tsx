import { observer } from "mobx-react-lite";
import LangButt from "./LangButt";
import langStore from "../store/langStore";
import styles from "./styles/translator.module.scss"
import { useEffect, useRef, useState } from "react";
import translateService from "../service/translateService";
import Loader from "./Loader";
import CopyButt from "./CopyButt";
import KeyboardButt from "./KeyboardButt";
import langService from "../service/langService";
import ClearButt from "./ClearButt";
import textStore from "../store/textStore";
import textService from "../service/textService";

const Translator = observer(() => {
    // const [input, setInput] = useState<string>("");
    // const [output, setOutput] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const maxLength = 5000; 
    const inputAreaRef = useRef<HTMLTextAreaElement>(null);
    const outputAreaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(()=>{
        fixHeights();
    }, [textStore.input, textStore.output])

    useEffect(() => {
        let debounceTimeout: ReturnType<typeof setTimeout>;

        const translate = async () => {
            setLoading(true);
            const fromLang = langStore.fromLang;
            const toLang = langStore.toLang;

            try {
                // const result = await translateService.translate(input, fromLang, toLang);
                // setOutput(result);
                const result = await translateService.translate(textStore.input, fromLang, toLang);
                textService.setOutput(result);
            } catch (error) {
                console.error('Translation error:', error);
                // setOutput("Error during translation");
                textService.setOutput("Error during translation");
            } finally {
                setLoading(false);
            }
        };

        if (textStore.input.length && !loading) {
            debounceTimeout = setTimeout(() => {
                translate();
            }, 300);  
        } else {
            textService.setOutput("");
        }

        return () => clearTimeout(debounceTimeout);
    }, [textStore.input, langStore.fromLang]);

    


    const fixHeights = () => {
        if (inputAreaRef.current && outputAreaRef.current) {
            inputAreaRef.current.style.height = 'auto'; 
            outputAreaRef.current.style.height = 'auto'; 
            const height = Math.max(inputAreaRef.current.scrollHeight, outputAreaRef.current.scrollHeight)
            inputAreaRef.current.style.height = `${height}px`; 
            outputAreaRef.current.style.height = `${height}px`;
        }
    }

    return(
        <div className={styles.translator} >
            
            <div className={styles.container}>
                <LangButt/>
                <div className={styles.field}>
                    <hr className={styles.field__hr }  style={{ marginTop: 0}}/>
                    <div className={`${styles.lang} ${styles.pcDisplay}`}>{langService.getNameByCode(langStore.fromLang)}</div>
                
                    <div className={`${styles.switcher} ${styles.tabletDisplay}`}>
                        <div className={styles.lang}>{langService.getNameByCode(langStore.fromLang)}</div>
                        <div className={styles.lang}>{langService.getNameByCode(langStore.toLang)}</div>

                    </div>
                    <hr className={styles.field__hr} style={{ marginRight: '35px'}}/>
                    <div className={styles.field__wrapper} >
                        <textarea
                            placeholder="Введите текст для перевода..."
                            value={textStore.input}
                            ref={inputAreaRef}
                            className={styles.field__text}
                            maxLength={maxLength}
                            onChange={e => textService.setInput(e.currentTarget.value || "")}
                        >
                           
                        </textarea>
                        {!!textStore.input && <ClearButt  tap={()=>{textService.setInput("")}}/>}
                        
                        <div className={styles.field__tools} >
                            
                            <div className={styles.field__tools_wrap}>
                                <KeyboardButt get={()=>textStore.input} set={(newInput:string)=>{textService.setInput(newInput)}} />
                            </div>
                            
                            <div className={styles.wordCount}>{maxLength-textStore.input.length}/{maxLength}</div>
                        </div>
                    </div>
                </div>



                <div className={styles.field} style={{backgroundColor: "var(--output-back)"}}>
                    <hr className={styles.field__hr}  style={{ marginTop: 0, opacity: 0}}/>
                    <div className={`${styles.lang} ${styles.pcDisplay}`}>{langService.getNameByCode(langStore.toLang)}</div>
                    <hr className={styles.field__hr} style={{ marginRight: '35px'}}/>
                    <div className={styles.field__wrapper} style={{paddingRight: "30px"}} >
                        <textarea
                            value={textStore.output}
                            ref={outputAreaRef}
                            className={`${styles.field__text} ${styles.field__text_readOnly}`}
                            readOnly
                            style={{backgroundColor: "transparent"}}
                        >
                        </textarea>
                        {loading && <Loader/>}
                        <div className={styles.field__tools} >
                            <div className={styles.field__tools_wrap}>
                                
                            </div>

                            <div className={styles.field__tools_wrap}>
                            {!!textStore.output && <CopyButt targetRef={outputAreaRef}/>}
                        
                                
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
});

export default Translator;