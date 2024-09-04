import { observer } from "mobx-react-lite";
import langService from "../service/langService";
import styles from "./styles/button.module.scss"
import toggle from "../assets/toggle.svg";

const LangButt = observer(() => {
    const tap = ()=>{
        langService.toggleLang();
    }    
    return(
        <div onClick={tap} className={styles.butt}>
            <img src={toggle} alt="<->" />
        </div>
    );
});

export default LangButt;